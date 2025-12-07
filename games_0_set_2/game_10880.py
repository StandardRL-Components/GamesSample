import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper classes for game objects
class GravityWell:
    def __init__(self, pos, screen_dims):
        self.pos = pygame.Vector2(pos)
        self.strength = 50.0  # Initial strength
        self.is_attractor = True
        self.max_strength = 250.0
        self.min_strength = 10.0
        self.color_attract = (100, 150, 255)
        self.color_repel = (255, 100, 100)

    def toggle_polarity(self):
        self.is_attractor = not self.is_attractor
        # sound effect: polarity_swap

    def change_strength(self, delta):
        self.strength = np.clip(self.strength + delta, self.min_strength, self.max_strength)

    def get_force_on(self, asteroid_pos):
        direction_vec = self.pos - asteroid_pos
        distance_sq = direction_vec.length_squared()

        if distance_sq < 25*25:  # singularity prevention
            distance_sq = 25*25

        force_magnitude = self.strength / distance_sq
        force_vec = direction_vec.normalize() * force_magnitude

        if not self.is_attractor:
            force_vec *= -1.5 # Repulsion is stronger to be more useful
        
        return force_vec

    def draw(self, surface, is_active):
        color = self.color_attract if self.is_attractor else self.color_repel
        
        # Base
        pygame.draw.circle(surface, (50, 50, 70), self.pos, 15)
        pygame.draw.circle(surface, color, self.pos, 12)
        
        # Strength indicator
        indicator_length = self.strength / self.max_strength * 30 + 5
        for i in range(1, 6):
            alpha = 255 - i * 40
            pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), int(15 + i * indicator_length/5), (*color, alpha))

        if is_active:
            # Glow effect for active well
            for i in range(4):
                glow_radius = 25 + i * 5
                glow_alpha = 100 - i * 25
                pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), glow_radius, (255, 255, 150, glow_alpha))

class Asteroid:
    def __init__(self, screen_dims):
        self.w, self.h = screen_dims
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            self.pos = pygame.Vector2(random.uniform(0, self.w), -10)
            self.vel = pygame.Vector2(random.uniform(-1, 1), random.uniform(0.5, 1.5))
        elif edge == 'bottom':
            self.pos = pygame.Vector2(random.uniform(0, self.w), self.h + 10)
            self.vel = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1.5, -0.5))
        elif edge == 'left':
            self.pos = pygame.Vector2(-10, random.uniform(0, self.h))
            self.vel = pygame.Vector2(random.uniform(0.5, 1.5), random.uniform(-1, 1))
        else: # right
            self.pos = pygame.Vector2(self.w + 10, random.uniform(0, self.h))
            self.vel = pygame.Vector2(random.uniform(-1.5, -0.5), random.uniform(-1, 1))
        
        self.radius = random.randint(4, 7)
        self.mass = self.radius**2 * 0.1
        self.color = (180, 180, 190)

    def update(self, total_force):
        acceleration = total_force / self.mass
        self.vel += acceleration
        # Speed limit
        speed = self.vel.length()
        if speed > 5:
            self.vel = self.vel.normalize() * 5
        self.pos += self.vel

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.pos.x), int(self.pos.y)), self.radius)
        pygame.draw.circle(surface, (220, 220, 230), (int(self.pos.x), int(self.pos.y)), self.radius, 1)

    def is_out_of_bounds(self):
        return not (-20 < self.pos.x < self.w + 20 and -20 < self.pos.y < self.h + 20)

class Particle:
    def __init__(self, pos, color):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(random.uniform(-2, 2), random.uniform(-2, 2))
        self.radius = random.uniform(1, 4)
        self.lifetime = random.uniform(20, 40) # in steps
        self.initial_lifetime = self.lifetime
        self.color = color

    def update(self):
        self.pos += self.vel
        self.vel *= 0.95 # friction
        self.lifetime -= 1

    def draw(self, surface):
        if self.lifetime > 0:
            alpha = int(255 * (self.lifetime / self.initial_lifetime))
            current_radius = int(self.radius * (self.lifetime / self.initial_lifetime))
            if current_radius > 0:
                # Using gfxdraw for alpha blending
                pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), current_radius, (*self.color, alpha))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Strategically manipulate gravity wells to guide asteroids into a central target zone. "
        "Master attraction and repulsion to score points before time runs out."
    )
    user_guide = (
        "Controls: Use ↑/↓ to change the active well's strength. Press space to cycle which well is active "
        "and shift to toggle its polarity between attract and repel."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.FPS = 60
        self.WIN_SCORE = 15
        self.MAX_TIME_SECONDS = 90
        self.MAX_STEPS = self.MAX_TIME_SECONDS * self.FPS
        self.MAX_ASTEROIDS = 25
        self.ASTEROID_SPAWN_INTERVAL = 20 # steps

        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_TARGET = (100, 255, 150)
        self.COLOR_TIMER_START = pygame.Color(255, 220, 0)
        self.COLOR_TIMER_END = pygame.Color(255, 50, 50)

        # Action mapping constants
        self.STRENGTH_DELTA = 15.0

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_gameover = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Define game zones
        self.target_zone = pygame.Rect(self.SCREEN_WIDTH // 2 - 50, self.SCREEN_HEIGHT // 2 - 50, 100, 100)
        
        # Initialize state variables
        self.gravity_wells = []
        self.asteroids = []
        self.particles = []
        self._was_space_held = False
        self._was_shift_held = False
        self.steps_since_last_spawn = 0
        self.active_well_index = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.steps_since_last_spawn = 0
        self.active_well_index = 0
        
        self.gravity_wells = [
            GravityWell((120, 100), (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)),
            GravityWell((self.SCREEN_WIDTH - 120, 100), (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)),
            GravityWell((120, self.SCREEN_HEIGHT - 100), (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)),
            GravityWell((self.SCREEN_WIDTH - 120, self.SCREEN_HEIGHT - 100), (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)),
        ]
        
        self.asteroids.clear()
        self.particles.clear()
        
        self._was_space_held = True # prevent press on first frame
        self._was_shift_held = True

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1

        # 1. Handle player input
        self._handle_actions(movement, space_held, shift_held)

        # 2. Update game logic
        self._spawn_asteroids()

        # Store previous asteroid distances for reward calculation
        prev_asteroid_distances = {id(ast): self._distance_to_target(ast) for ast in self.asteroids}
        
        scored_this_step = self._update_asteroids()
        self._update_particles()
        
        # 3. Calculate reward
        reward += self._calculate_reward(prev_asteroid_distances, scored_this_step)
        
        # 4. Check for termination
        terminated = self._check_termination()
        if terminated:
            if self.win:
                reward += 100 # Win bonus
            else:
                reward -= 50 # Timeout penalty
        truncated = False # This environment does not truncate based on step count alone

        # 5. Update input history for next step
        self._was_space_held = space_held
        self._was_shift_held = shift_held

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_actions(self, movement, space_held, shift_held):
        space_pressed = space_held and not self._was_space_held
        shift_pressed = shift_held and not self._was_shift_held

        if space_pressed:
            # Cycle through which well is active
            self.active_well_index = (self.active_well_index + 1) % len(self.gravity_wells)
            # sound effect: UI_focus_change

        active_well = self.gravity_wells[self.active_well_index]
        
        if shift_pressed:
            # Toggle active well's polarity
            active_well.toggle_polarity()

        # Modify active well's strength with up/down
        if movement == 1: # Up
            active_well.change_strength(self.STRENGTH_DELTA)
        elif movement == 2: # Down
            active_well.change_strength(-self.STRENGTH_DELTA)
        # Left/Right actions are ignored as per the adapted control scheme
        # A more complex scheme could use them to change well angle/direction if they had one

    def _spawn_asteroids(self):
        self.steps_since_last_spawn += 1
        if self.steps_since_last_spawn >= self.ASTEROID_SPAWN_INTERVAL and len(self.asteroids) < self.MAX_ASTEROIDS:
            self.asteroids.append(Asteroid((self.SCREEN_WIDTH, self.SCREEN_HEIGHT)))
            self.steps_since_last_spawn = 0
    
    def _update_asteroids(self):
        asteroids_to_remove = []
        scored_this_step = 0
        for i, asteroid in enumerate(self.asteroids):
            total_force = pygame.Vector2(0, 0)
            for well in self.gravity_wells:
                total_force += well.get_force_on(asteroid.pos)
            
            asteroid.update(total_force)

            if asteroid.is_out_of_bounds():
                asteroids_to_remove.append(asteroid)
            elif self.target_zone.collidepoint(asteroid.pos):
                self.score += 1
                scored_this_step += 1
                asteroids_to_remove.append(asteroid)
                # sound effect: score_point
                for _ in range(20): # Create particle explosion
                    self.particles.append(Particle(asteroid.pos, self.COLOR_TARGET))
        
        self.asteroids = [ast for ast in self.asteroids if ast not in asteroids_to_remove]
        return scored_this_step

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifetime > 0]
        for p in self.particles:
            p.update()

    def _calculate_reward(self, prev_distances, scored_this_step):
        reward = 0
        # Event-based reward for scoring
        reward += scored_this_step * 10.0

        # Continuous reward for moving towards target
        for ast in self.asteroids:
            prev_dist = prev_distances.get(id(ast))
            if prev_dist is not None:
                new_dist = self._distance_to_target(ast)
                if new_dist < prev_dist:
                    reward += 0.01 # Getting closer
                else:
                    reward -= 0.001 # Getting further

        return reward
    
    def _distance_to_target(self, asteroid):
        return asteroid.pos.distance_to(self.target_zone.center)

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            self.game_over = True
            self.win = True
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = False
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": (self.MAX_STEPS - self.steps) / self.FPS,
            "win": self.win,
        }

    def _render_game(self):
        # Render target zone with glow
        for i in range(10):
            alpha = 30 - i * 3
            glow_rect = self.target_zone.inflate(i * 4, i * 4)
            shape_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, (*self.COLOR_TARGET, alpha), (0, 0, *glow_rect.size), border_radius=15)
            self.screen.blit(shape_surf, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_TARGET, self.target_zone, 2, border_radius=10)

        # Render gravity wells
        for i, well in enumerate(self.gravity_wells):
            well.draw(self.screen, i == self.active_well_index)
        
        # Render asteroids
        for asteroid in self.asteroids:
            asteroid.draw(self.screen)

        # Render particles
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        for particle in self.particles:
            particle.draw(s)
        self.screen.blit(s, (0, 0))

    def _render_ui(self):
        # Score display
        score_text = self.font_main.render(f"SCORE: {self.score}/{self.WIN_SCORE}", True, (200, 200, 220))
        self.screen.blit(score_text, (10, 10))

        # Timer bar
        time_ratio = max(0, (self.MAX_STEPS - self.steps) / self.MAX_STEPS)
        timer_width = int((self.SCREEN_WIDTH - 20) * time_ratio)
        timer_color = self.COLOR_TIMER_START.lerp(self.COLOR_TIMER_END, 1 - time_ratio)
        
        pygame.draw.rect(self.screen, (50, 50, 70), (10, 40, self.SCREEN_WIDTH - 20, 10))
        if timer_width > 0:
            pygame.draw.rect(self.screen, timer_color, (10, 40, timer_width, 10))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "MISSION COMPLETE" if self.win else "TIME UP"
            color = self.COLOR_TARGET if self.win else self.COLOR_TIMER_END
            
            text_surface = self.font_gameover.render(message, True, color)
            text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surface, text_rect)

    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    # Un-comment the line below to run with a display window
    # os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    
    env = GameEnv()
    
    # --- Manual Play ---
    # Controls:
    # Arrow Up/Down: Increase/Decrease strength of active well
    # Space: Cycle which gravity well is active (highlighted)
    # Shift: Toggle active well between Attract (blue) and Repel (red)
    
    obs, info = env.reset()
    terminated = False
    
    # Pygame window for human interaction
    # This part requires a display driver.
    try:
        display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Gravity Well")
        running = True
    except pygame.error:
        print("Pygame display could not be initialized. Running in headless mode.")
        running = False # Cannot run interactive loop without display
    
    while running:
        movement = 0 # none
        space_bar = 0
        shift_key = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        # Left/Right are part of the action space but unused in this control scheme
        # elif keys[pygame.K_LEFT]: movement = 3
        # elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_bar = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_key = 1
        
        action = [movement, space_bar, shift_key]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Blit the observation from the env to the display screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over. Final Score: {info['score']}, Win: {info['win']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()

        env.clock.tick(env.FPS)

    env.close()