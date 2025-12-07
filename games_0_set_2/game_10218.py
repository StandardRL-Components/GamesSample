import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:06:13.252355
# Source Brief: brief_00218.md
# Brief Index: 218
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# --- Helper Classes for Game Entities ---

class Player:
    def __init__(self, pos, radius):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(0, 0)
        self.radius = radius
        self.color = (0, 255, 128) # Bright Green

class Antibody:
    def __init__(self, path, radius, speed):
        self.path = [pygame.Vector2(p) for p in path]
        self.path_index = 0
        self.pos = pygame.Vector2(self.path[0])
        self.radius = radius
        self.speed = speed
        self.color = (255, 50, 50) # Bright Red
        self.direction = 1

    def update(self, base_speed):
        target = self.path[self.path_index]
        move_vec = target - self.pos
        
        if move_vec.length_squared() < (self.speed * base_speed)**2:
            self.path_index = (self.path_index + self.direction) % len(self.path)
            # Reverse direction at ends of path
            if self.path_index == 0 or self.path_index == len(self.path) - 1:
                self.direction *= -1
        else:
            self.pos += move_vec.normalize() * self.speed * base_speed

class Particle:
    def __init__(self, pos, vel, radius, lifespan):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.radius = radius
        self.color = (100, 150, 255) # Bright Blue
        self.lifespan = lifespan
        self.trail = []

    def update(self):
        self.lifespan -= 1
        self.trail.append(self.pos.copy())
        if len(self.trail) > 5:
            self.trail.pop(0)
        self.pos += self.vel

class Organelle:
    def __init__(self, pos, radius):
        self.pos = pygame.Vector2(pos)
        self.radius = radius
        self.color = (60, 60, 90) # Dark Grey/Blue

# --- Main Gymnasium Environment ---

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a virus through a hostile cellular environment, injecting organelles to "
        "replicate and ultimately infect the nucleus."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to inject viral particles. "
        "Press shift to flip gravity."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    
    COLOR_BG = (15, 20, 30)
    COLOR_NUCLEUS = (138, 43, 226)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_ABILITY_UNLOCKED = (255, 215, 0)

    PLAYER_RADIUS = 12
    PLAYER_ACCEL = 0.8
    PLAYER_FRICTION = 0.92
    
    GRAVITY_STRENGTH = 0.4
    
    PARTICLE_SPEED = 8
    PARTICLE_RADIUS = 4
    PARTICLE_LIFESPAN = 80
    PARTICLE_COOLDOWN = 10
    
    WIN_CONDITION_INJECTIONS = 50
    MAX_EPISODE_STEPS = 5000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_popup = pygame.font.SysFont("Arial", 28, bold=True)
        
        self.render_mode = render_mode

        # Initialize state variables
        self.player = None
        self.antibodies = []
        self.particles = []
        self.organelles = []
        self.nucleus = None
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.gravity_direction = 1
        self.particle_cooldown_timer = 0
        self.viral_particle_count = 0
        self.successful_injections = 0
        self.antibody_base_speed = 1.0
        
        self.unlocked_abilities = set()
        self.player_accel_modifier = 1.0
        self.particle_cooldown_modifier = 1.0
        
        self.last_move_direction = pygame.Vector2(0, -1)
        self.shift_pressed_last_frame = False
        
        self.popup_text = None
        self.popup_timer = 0

        # This call is not strictly necessary in the final version,
        # but useful during development.
        # self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player = Player(pos=(self.WIDTH // 2, 50), radius=self.PLAYER_RADIUS)
        
        self.antibodies = []
        self.particles = []
        
        self.organelles = [
            Organelle(pos=(100, 200), radius=40),
            Organelle(pos=(540, 200), radius=40),
            Organelle(pos=(320, 120), radius=30),
            Organelle(pos=(320, 280), radius=30),
        ]
        
        self.nucleus = Organelle(pos=(self.WIDTH // 2, self.HEIGHT // 2), radius=50)
        self.nucleus.color = self.COLOR_NUCLEUS
        
        self._setup_antibodies()
        
        self.gravity_direction = 1
        self.particle_cooldown_timer = 0
        self.viral_particle_count = 10
        self.successful_injections = 0
        self.antibody_base_speed = 1.0
        
        self.unlocked_abilities = set()
        self.player_accel_modifier = 1.0
        self.particle_cooldown_modifier = 1.0
        
        self.last_move_direction = pygame.Vector2(0, -1)
        self.shift_pressed_last_frame = False
        
        self.popup_text = None
        self.popup_timer = 0
        
        return self._get_observation(), self._get_info()

    def _setup_antibodies(self):
        self.antibodies = [
            Antibody(path=[(50, 80), (self.WIDTH - 50, 80)], radius=10, speed=1.2),
            Antibody(path=[(50, 320), (self.WIDTH - 50, 320)], radius=10, speed=1.2),
            Antibody(path=[(180, 50), (180, self.HEIGHT - 50)], radius=8, speed=1.0),
            Antibody(path=[(460, 50), (460, self.HEIGHT - 50)], radius=8, speed=1.0),
        ]

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        
        if self.game_over:
            return self._get_observation(), reward, True, False, self._get_info()

        self.steps += 1
        reward += 0.01 # Small reward for surviving

        self._handle_input(action)
        self._update_game_state()
        
        # --- Check Collisions and Update Rewards ---
        collision_reward, terminated_by_collision = self._check_collisions()
        reward += collision_reward
        if terminated_by_collision:
            self.game_over = True

        # --- Check Win/Loss Conditions ---
        if self.successful_injections >= self.WIN_CONDITION_INJECTIONS:
            reward += 100
            terminated = True
            self.game_over = True
            self._show_popup("NUCLEUS INFECTED!", 90)
        elif self.game_over:
            reward -= 100
            terminated = True
        elif self.steps >= self.MAX_EPISODE_STEPS:
            truncated = True
            self.game_over = True

        # Update score
        self.score += reward
            
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1

        # --- Movement ---
        move_force = pygame.Vector2(0, 0)
        if movement == 1: # Up
            move_force.y = -1
        elif movement == 2: # Down
            move_force.y = 1
        elif movement == 3: # Left
            move_force.x = -1
        elif movement == 4: # Right
            move_force.x = 1
        
        if move_force.length_squared() > 0:
            self.player.vel += move_force.normalize() * self.PLAYER_ACCEL * self.player_accel_modifier
            self.last_move_direction = move_force.normalize()

        # --- Inject Particle (Space) ---
        if space_pressed and self.particle_cooldown_timer == 0 and self.viral_particle_count > 0:
            # sfx: player_shoot.wav
            self.viral_particle_count -= 1
            particle_vel = self.last_move_direction * self.PARTICLE_SPEED
            new_particle = Particle(self.player.pos.copy(), particle_vel, self.PARTICLE_RADIUS, self.PARTICLE_LIFESPAN)
            self.particles.append(new_particle)
            self.particle_cooldown_timer = self.PARTICLE_COOLDOWN * self.particle_cooldown_modifier

        # --- Flip Gravity (Shift) ---
        if shift_pressed and not self.shift_pressed_last_frame:
            # sfx: gravity_flip.wav
            self.gravity_direction *= -1
        self.shift_pressed_last_frame = shift_pressed

    def _update_game_state(self):
        # Update timers
        if self.particle_cooldown_timer > 0:
            self.particle_cooldown_timer -= 1
        if self.popup_timer > 0:
            self.popup_timer -= 1

        # Update player
        self.player.vel.y += self.gravity_direction * self.GRAVITY_STRENGTH
        self.player.vel *= self.PLAYER_FRICTION
        self.player.pos += self.player.vel
        
        # Player boundary constraints
        self.player.pos.x = np.clip(self.player.pos.x, self.player.radius, self.WIDTH - self.player.radius)
        self.player.pos.y = np.clip(self.player.pos.y, self.player.radius, self.HEIGHT - self.player.radius)

        # Update antibodies
        if self.steps % 500 == 0 and self.steps > 0:
            self.antibody_base_speed += 0.05
        for ab in self.antibodies:
            ab.update(self.antibody_base_speed)
            
        # Update particles
        for p in self.particles[:]:
            p.update()
            if p.lifespan <= 0:
                self.particles.remove(p)

    def _check_collisions(self):
        reward = 0
        terminated = False

        # Player vs Antibodies
        for ab in self.antibodies:
            if self.player.pos.distance_to(ab.pos) < self.player.radius + ab.radius:
                # sfx: player_death.wav
                terminated = True
                return reward, terminated

        # Particles vs Organelles/Nucleus/Walls
        for p in self.particles[:]:
            # vs Walls
            if not (0 < p.pos.x < self.WIDTH and 0 < p.pos.y < self.HEIGHT):
                self.particles.remove(p)
                continue

            # vs Nucleus
            if p.pos.distance_to(self.nucleus.pos) < p.radius + self.nucleus.radius:
                self.particles.remove(p)
                continue
            
            # vs Organelles
            for organelle in self.organelles:
                if p.pos.distance_to(organelle.pos) < p.radius + organelle.radius:
                    # sfx: particle_replicate.wav
                    self.particles.remove(p)
                    self.successful_injections += 1
                    self.viral_particle_count = min(99, self.viral_particle_count + 2)
                    reward += 1
                    
                    # Check for ability unlocks
                    if self.successful_injections == 20 and "faster_movement" not in self.unlocked_abilities:
                        self.unlocked_abilities.add("faster_movement")
                        self.player_accel_modifier = 1.3
                        reward += 5
                        self._show_popup("ABILITY: FASTER MOVEMENT!", 60)
                    elif self.successful_injections == 40 and "faster_injection" not in self.unlocked_abilities:
                        self.unlocked_abilities.add("faster_injection")
                        self.particle_cooldown_modifier = 0.5
                        reward += 5
                        self._show_popup("ABILITY: RAPID INJECTION!", 60)
                    break
        
        return reward, terminated
    
    def _show_popup(self, text, duration):
        self.popup_text = text
        self.popup_timer = duration

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
            "viral_particles": self.viral_particle_count,
            "successful_injections": self.successful_injections,
        }

    def _render_game(self):
        # Draw background details
        for _ in range(10):
            pygame.gfxdraw.filled_circle(
                self.screen,
                self.np_random.integers(0, self.WIDTH),
                self.np_random.integers(0, self.HEIGHT),
                self.np_random.integers(10, 50),
                (20, 25, 40)
            )

        # Draw Nucleus
        pygame.gfxdraw.filled_circle(self.screen, int(self.nucleus.pos.x), int(self.nucleus.pos.y), self.nucleus.radius, self.nucleus.color)
        pygame.gfxdraw.aacircle(self.screen, int(self.nucleus.pos.x), int(self.nucleus.pos.y), self.nucleus.radius, self.nucleus.color)

        # Draw Organelles
        for o in self.organelles:
            pygame.gfxdraw.filled_circle(self.screen, int(o.pos.x), int(o.pos.y), o.radius, o.color)
            pygame.gfxdraw.aacircle(self.screen, int(o.pos.x), int(o.pos.y), o.radius, o.color)

        # Draw Particles
        for p in self.particles:
            if len(p.trail) > 1:
                for i, pos in enumerate(p.trail):
                    alpha = int(255 * (i / len(p.trail)))
                    color = (*p.color, alpha)
                    pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), p.radius-2, color)
            pygame.gfxdraw.filled_circle(self.screen, int(p.pos.x), int(p.pos.y), p.radius, p.color)
            pygame.gfxdraw.aacircle(self.screen, int(p.pos.x), int(p.pos.y), p.radius, p.color)
            
        # Draw Antibodies
        for ab in self.antibodies:
            pygame.gfxdraw.filled_circle(self.screen, int(ab.pos.x), int(ab.pos.y), ab.radius, ab.color)
            pygame.gfxdraw.aacircle(self.screen, int(ab.pos.x), int(ab.pos.y), ab.radius, ab.color)

        # Draw Player
        if not self.game_over:
            # Glow effect
            for i in range(4):
                alpha = 60 - i * 15
                radius = self.player.radius + i * 3
                color = (*self.player.color, alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(self.player.pos.x), int(self.player.pos.y), radius, color)
            
            pygame.gfxdraw.filled_circle(self.screen, int(self.player.pos.x), int(self.player.pos.y), self.player.radius, self.player.color)
            pygame.gfxdraw.aacircle(self.screen, int(self.player.pos.x), int(self.player.pos.y), self.player.radius, self.player.color)

    def _render_ui(self):
        # --- Draw UI Text ---
        # Viral Particles
        particle_text = self.font_ui.render(f"PARTICLES: {self.viral_particle_count}", True, self.COLOR_UI_TEXT)
        self.screen.blit(particle_text, (10, 10))
        
        # Injections
        injection_text = self.font_ui.render(f"INJECTIONS: {self.successful_injections}/{self.WIN_CONDITION_INJECTIONS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(injection_text, (10, 35))

        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(centerx=self.WIDTH / 2, y=10)
        self.screen.blit(score_text, score_rect)
        
        # --- Draw Gravity Indicator ---
        arrow_points = []
        if self.gravity_direction == 1: # Down
            arrow_points = [(self.WIDTH - 25, self.HEIGHT / 2 - 10), (self.WIDTH - 15, self.HEIGHT / 2 - 10), (self.WIDTH-20, self.HEIGHT/2)]
        else: # Up
            arrow_points = [(self.WIDTH - 25, self.HEIGHT / 2 + 10), (self.WIDTH - 15, self.HEIGHT / 2 + 10), (self.WIDTH-20, self.HEIGHT/2)]
        pygame.gfxdraw.aapolygon(self.screen, arrow_points, self.COLOR_UI_TEXT)
        pygame.gfxdraw.filled_polygon(self.screen, arrow_points, self.COLOR_UI_TEXT)

        # --- Draw Popup Text ---
        if self.popup_timer > 0:
            color = self.COLOR_ABILITY_UNLOCKED
            if "INFECTED" in self.popup_text:
                color = (100, 255, 100) # Green for win
            
            popup_surf = self.font_popup.render(self.popup_text, True, color)
            popup_rect = popup_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 100))
            
            # Fade effect
            alpha = min(255, int(255 * (self.popup_timer / 30)))
            popup_surf.set_alpha(alpha)
            
            self.screen.blit(popup_surf, popup_rect)

    def close(self):
        pygame.quit()
        

if __name__ == '__main__':
    # --- Manual Play Code ---
    # Re-enable display for manual play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Viral Infiltration")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- RESETTING ---")
                obs, info = env.reset()
                total_reward = 0

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            pygame.time.wait(2000) # Pause for 2 seconds before resetting
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)
        
    env.close()