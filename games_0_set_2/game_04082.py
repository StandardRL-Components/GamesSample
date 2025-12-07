import os
import os
import pygame

os.environ["SDL_VIDEODRIVER"] = "dummy"

import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random


class GameEnv(gym.Env):
    """
    A retro-style, top-down arcade space shooter Gymnasium environment.
    The player must defeat waves of descending aliens while dodging their fire.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move. Hold Shift for a temporary shield. Press Space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defeat waves of descending aliens in this fast-paced arcade shooter. Use your shield wisely and aim for a high score!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        """
        Initializes the game environment.
        """
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Colors
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_BULLET = (100, 255, 100)
        self.COLOR_ALIEN = (255, 50, 150)
        self.COLOR_ALIEN_BULLET = (255, 50, 50)
        self.COLOR_SHIELD = (100, 200, 255, 100)
        self.COLOR_TEXT = (255, 255, 255)
        self.EXPLOSION_COLORS = [(255, 255, 0), (255, 150, 0), (255, 50, 0)]

        # Game constants
        self.MAX_STEPS = 5000
        self.TOTAL_ALIENS_TO_SPAWN = 50
        self.PLAYER_SPEED = 6
        self.PLAYER_BULLET_SPEED = 10
        self.PLAYER_FIRE_COOLDOWN = 5  # frames
        self.SHIELD_DURATION = 5
        self.SHIELD_COOLDOWN = 30
        self.NEAR_MISS_DISTANCE = 30
        self.MAX_PROJECTILES = 100

        # Initialize state variables to prevent attribute errors
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None
        self.player_pos = [0, 0]
        self.player_lives = 0
        self.player_fire_timer = 0
        self.shield_active = False
        self.shield_timer = 0
        self.shield_cooldown_timer = 0
        self.player_bullets = []
        self.alien_bullets = []
        self.aliens = []
        self.particles = []
        self.stars = []
        self.wave_number = 0
        self.aliens_defeated = 0
        self.wave_alien_speed = 1.0
        self.wave_alien_fire_rate = 0.1
        
        # self.reset() is called by the test harness, no need to call it here.
        # self.validate_implementation() is for debugging and not needed for the final env.

    def reset(self, seed=None, options=None):
        """
        Resets the game to its initial state.
        """
        super().reset(seed=seed)
        if self.np_random is None:
             self.np_random, seed = gym.utils.seeding.np_random(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        # Player state
        self.player_pos = [self.WIDTH // 2, self.HEIGHT - 50]
        self.player_lives = 3
        self.player_fire_timer = 0
        
        # Shield state
        self.shield_active = False
        self.shield_timer = 0
        self.shield_cooldown_timer = 0

        # Entity lists
        self.player_bullets = []
        self.alien_bullets = []
        self.aliens = []
        self.particles = []

        # Wave management
        self.wave_number = 0
        self.aliens_defeated = 0
        self.wave_alien_speed = 1.0
        self.wave_alien_fire_rate = 0.1
        self._spawn_wave()

        # Visuals
        self.stars = [
            (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.integers(1, 4))
            for _ in range(100)
        ]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        """
        Advances the game state by one frame.
        """
        if self.auto_advance:
            self.clock.tick(30)

        reward = 0.1  # Survival reward
        
        # --- 1. ACTION HANDLING ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_player_input(movement, space_held, shift_held)
        
        if shift_held and self.shield_cooldown_timer == 0 and not self.shield_active:
            self.shield_active = True
            self.shield_timer = self.SHIELD_DURATION
            self.shield_cooldown_timer = self.SHIELD_COOLDOWN + self.SHIELD_DURATION
            # Penalty for using shield when no bullets are near
            if not any(math.hypot(b[0] - self.player_pos[0], b[1] - self.player_pos[1]) < 100 for b in self.alien_bullets):
                reward -= 0.2

        # --- 2. UPDATE GAME STATE ---
        self._update_timers()
        self._update_bullets()
        self._update_aliens()
        self._update_particles()
        self._update_starfield()

        # --- 3. COLLISION DETECTION & REWARDS ---
        reward += self._handle_collisions()
        
        # --- 4. WAVE MANAGEMENT ---
        if not self.aliens and self.aliens_defeated < self.TOTAL_ALIENS_TO_SPAWN:
            self._spawn_wave()
        
        # --- 5. TERMINATION CHECK ---
        self.steps += 1
        win_condition = self.aliens_defeated >= self.TOTAL_ALIENS_TO_SPAWN
        lose_condition = self.player_lives <= 0
        max_steps_reached = self.steps >= self.MAX_STEPS
        
        terminated = win_condition or lose_condition
        truncated = max_steps_reached
        
        if win_condition:
            reward += 100 # Win bonus
        if lose_condition:
            reward -= 100 # Lose penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_player_input(self, movement, space_held, shift_held):
        """Handles player movement and firing."""
        # Movement
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2: self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3: self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4: self.player_pos[0] += self.PLAYER_SPEED
        
        self.player_pos[0] = np.clip(self.player_pos[0], 10, self.WIDTH - 10)
        self.player_pos[1] = np.clip(self.player_pos[1], 10, self.HEIGHT - 10)

        # Firing
        if space_held and self.player_fire_timer == 0 and len(self.player_bullets) < self.MAX_PROJECTILES:
            # sfx: player_shoot.wav
            self.player_bullets.append(list(self.player_pos))
            self.player_fire_timer = self.PLAYER_FIRE_COOLDOWN

    def _update_timers(self):
        """Updates all cooldowns and timers."""
        if self.player_fire_timer > 0:
            self.player_fire_timer -= 1
        if self.shield_timer > 0:
            self.shield_timer -= 1
            if self.shield_timer == 0:
                self.shield_active = False
        if self.shield_cooldown_timer > 0:
            self.shield_cooldown_timer -= 1
            
    def _update_bullets(self):
        """Moves player and alien bullets and removes off-screen ones."""
        self.player_bullets = [[b[0], b[1] - self.PLAYER_BULLET_SPEED] for b in self.player_bullets if b[1] > 0]
        # FIX: Preserve the bullet's speed (b[2]) in the new list.
        self.alien_bullets = [[b[0], b[1] + b[2], b[2]] for b in self.alien_bullets if b[1] < self.HEIGHT]

    def _update_aliens(self):
        """Moves aliens and handles their firing."""
        for alien in self.aliens:
            # Sinusoidal movement
            alien['pos'][0] = alien['origin_x'] + math.sin(self.steps * 0.05 + alien['phase']) * alien['amplitude']
            alien['pos'][1] += self.wave_alien_speed
            
            # Firing
            if self.np_random.random() < self.wave_alien_fire_rate and len(self.alien_bullets) < self.MAX_PROJECTILES:
                # sfx: alien_shoot.wav
                self.alien_bullets.append([alien['pos'][0], alien['pos'][1], 4]) # x, y, speed

            # Alien reaches bottom
            if alien['pos'][1] > self.HEIGHT:
                self.player_lives -= 1
                self._create_explosion(self.player_pos, 40)
                alien['health'] = 0 # Mark for removal
                if self.player_lives > 0:
                    self.player_pos = [self.WIDTH // 2, self.HEIGHT - 50] # Reset position

        self.aliens = [a for a in self.aliens if a['health'] > 0]

    def _update_particles(self):
        """Updates position and lifetime of particles."""
        self.particles = [p for p in self.particles if p[3] > 0]
        for p in self.particles:
            p[0] += p[2][0]
            p[1] += p[2][1]
            p[3] -= 0.1 # Lifetime

    def _update_starfield(self):
        """Updates star positions for parallax effect."""
        for i, (x, y, speed) in enumerate(self.stars):
            y_new = (y + speed) % self.HEIGHT
            if y_new < y: # Star wrapped around
                x_new = self.np_random.integers(0, self.WIDTH)
            else:
                x_new = x
            self.stars[i] = (x_new, y_new, speed)

    def _handle_collisions(self):
        """Checks for and handles all game collisions."""
        reward = 0
        
        # Player bullets vs aliens
        aliens_to_remove = []
        bullets_to_remove = []
        for b_idx, bullet in enumerate(self.player_bullets):
            for a_idx, alien in enumerate(self.aliens):
                if alien in aliens_to_remove: continue
                if math.hypot(bullet[0] - alien['pos'][0], bullet[1] - alien['pos'][1]) < 15:
                    self._create_explosion(alien['pos'], 20)
                    reward += 10
                    self.score += 100
                    aliens_to_remove.append(alien)
                    self.aliens_defeated += 1
                    bullets_to_remove.append(bullet)
                    break
        self.player_bullets = [b for b in self.player_bullets if b not in bullets_to_remove]
        
        # Alien bullets vs player
        bullets_to_remove = []
        for bullet in self.alien_bullets:
            dist = math.hypot(bullet[0] - self.player_pos[0], bullet[1] - self.player_pos[1])
            if dist < 15: # Hit
                if self.shield_active:
                    self._create_explosion(bullet, 10, count=5)
                else:
                    self.player_lives -= 1
                    self._create_explosion(self.player_pos, 40)
                    if self.player_lives > 0:
                        self.player_pos = [self.WIDTH // 2, self.HEIGHT - 50] # Reset
                bullets_to_remove.append(bullet)
            elif dist < self.NEAR_MISS_DISTANCE:
                reward += 5 # Near miss reward
        self.alien_bullets = [b for b in self.alien_bullets if b not in bullets_to_remove]

        # Player vs aliens
        for alien in self.aliens:
            if alien in aliens_to_remove: continue
            if math.hypot(self.player_pos[0] - alien['pos'][0], self.player_pos[1] - alien['pos'][1]) < 20:
                if not self.shield_active:
                    self.player_lives -= 1
                    self._create_explosion(self.player_pos, 40)
                    aliens_to_remove.append(alien)
                    if self.player_lives > 0:
                        self.player_pos = [self.WIDTH // 2, self.HEIGHT - 50] # Reset

        self.aliens = [a for a in self.aliens if a not in aliens_to_remove]
        return reward

    def _spawn_wave(self):
        """Spawns a new wave of aliens with increased difficulty."""
        self.wave_number += 1
        self.wave_alien_speed = min(3.0, 1.0 + (self.wave_number - 1) * 0.1)
        self.wave_alien_fire_rate = min(0.5, 0.02 + (self.wave_number - 1) * 0.02)
        
        aliens_to_spawn = min(10, self.TOTAL_ALIENS_TO_SPAWN - self.aliens_defeated)
        for i in range(aliens_to_spawn):
            self.aliens.append({
                'pos': [self.np_random.integers(50, self.WIDTH - 50), -20 - i * 30],
                'origin_x': self.np_random.integers(100, self.WIDTH - 100),
                'amplitude': self.np_random.integers(50, 150),
                'phase': self.np_random.random() * 2 * math.pi,
                'health': 1,
            })

    def _create_explosion(self, pos, size, count=30):
        """Creates particles for an explosion effect."""
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * (size / 10.0)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.random() * 1.5 + 0.5
            color_idx = self.np_random.integers(len(self.EXPLOSION_COLORS))
            self.particles.append([pos[0], pos[1], vel, lifetime, self.EXPLOSION_COLORS[color_idx]])

    def _get_observation(self):
        """Renders the current game state to the screen and returns it as a numpy array."""
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders all non-UI game elements."""
        # Stars
        for x, y, speed in self.stars:
            color = int(80 * (speed / 3.0))
            self.screen.set_at((int(x), int(y)), (color, color, color))
            
        # Alien Bullets
        for x, y, _ in self.alien_bullets:
            pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), 4, self.COLOR_ALIEN_BULLET)
            pygame.gfxdraw.aacircle(self.screen, int(x), int(y), 4, self.COLOR_ALIEN_BULLET)

        # Player Bullets
        for x, y in self.player_bullets:
            pygame.draw.line(self.screen, self.COLOR_PLAYER_BULLET, (x, y), (x, y+5), 2)

        # Aliens
        for alien in self.aliens:
            x, y = int(alien['pos'][0]), int(alien['pos'][1])
            points = [(x, y - 8), (x - 8, y + 8), (x + 8, y + 8)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ALIEN)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ALIEN)

        # Particles
        for x, y, vel, life, color in self.particles:
            size = max(0, int(life * 2))
            if size > 0:
                pygame.draw.circle(self.screen, color, (int(x), int(y)), size)

        # Player
        px, py = int(self.player_pos[0]), int(self.player_pos[1])
        player_points = [(px, py - 12), (px - 10, py + 10), (px + 10, py + 10)]
        pygame.gfxdraw.aapolygon(self.screen, player_points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, player_points, self.COLOR_PLAYER)
        
        # Player engine trail
        engine_y = py + 12
        pygame.draw.circle(self.screen, (255, 200, 0), (px, engine_y), 4)
        pygame.draw.circle(self.screen, (255, 100, 0), (px, engine_y), 2)

        # Shield
        if self.shield_active:
            alpha = 50 + int(150 * (self.shield_timer / self.SHIELD_DURATION))
            color = (*self.COLOR_SHIELD[:3], alpha)
            radius = 25
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius, color)
            pygame.gfxdraw.aacircle(temp_surf, radius, radius, radius, (200, 220, 255, alpha))
            self.screen.blit(temp_surf, (px - radius, py - radius))

    def _render_ui(self):
        """Renders the UI elements."""
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        wave_text = self.font_small.render(f"WAVE: {self.wave_number}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 40))

        # Lives display
        for i in range(self.player_lives):
            px, py = self.WIDTH - 30 - i * 25, 25
            points = [(px, py - 8), (px - 7, py + 7), (px + 7, py + 7)]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _get_info(self):
        """Returns a dictionary with game information."""
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "wave": self.wave_number,
            "aliens_defeated": self.aliens_defeated,
        }

    def close(self):
        """Closes the Pygame window."""
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    # The main __init__ sets the dummy driver, but for the interactive
    # block, we need to unset it to create a real window.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    # --- To display the game in a window ---
    pygame.display.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Arcade Shooter")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    # Game loop
    while not done:
        # Simple human input mapping
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30) # Match env's framerate

    print(f"Game Over! Final Info: {info}")
    env.close()