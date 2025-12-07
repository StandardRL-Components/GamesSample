
# Generated: 2025-08-28T00:05:48.548674
# Source Brief: brief_03686.md
# Brief Index: 3686

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Press space to jump over the cacti. Survive as long as you can!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A pixel-art dinosaur must leap over procedurally generated cacti in a desert landscape. The game speeds up over time. Survive for as long as possible to maximize your score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 2000

        # Colors
        self.COLOR_SKY = (135, 206, 235)
        self.COLOR_GROUND = (210, 180, 140)
        self.COLOR_DINO = (50, 50, 50)
        self.COLOR_CACTUS = (0, 128, 0)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_HEART = (255, 0, 0)
        self.COLOR_HIT_FLASH = (255, 100, 100, 150)
        self.COLOR_DUST = (181, 153, 120)

        # Physics & Game Mechanics
        self.GRAVITY = 1.8
        self.JUMP_STRENGTH = -25
        self.GROUND_Y = self.HEIGHT - 80
        self.INITIAL_LIVES = 3

        # Dino properties
        self.DINO_X = 80
        self.DINO_WIDTH = 40
        self.DINO_HEIGHT = 44

        # Difficulty scaling
        self.INITIAL_CACTUS_SPEED = 8
        self.INITIAL_CACTUS_SPAWN_PROB = 1 / (self.FPS * 1.5) # ~ every 1.5s
        self.SPEED_INCREASE_INTERVAL = 100
        self.SPEED_INCREASE_AMOUNT = 0.5
        self.SPAWN_PROB_INCREASE_AMOUNT = 0.002

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_heart = pygame.font.SysFont("sans", 30)

        # --- State Variables ---
        self.dino_y = 0
        self.dino_vy = 0
        self.is_jumping = False
        self.cacti = []
        self.particles = []
        self.lives = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cactus_speed = 0
        self.cactus_spawn_prob = 0
        self.invincibility_timer = 0
        self.hit_flash_timer = 0
        self.space_was_held = False # To detect rising edge of space press

        # Initialize state
        self.reset()
        
        # Run self-check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.dino_y = self.GROUND_Y
        self.dino_vy = 0
        self.is_jumping = False
        self.cacti = []
        self.particles = []
        self.lives = self.INITIAL_LIVES
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.invincibility_timer = 0
        self.hit_flash_timer = 0
        self.space_was_held = False

        # Reset difficulty
        self.cactus_speed = self.INITIAL_CACTUS_SPEED
        self.cactus_spawn_prob = self.INITIAL_CACTUS_SPAWN_PROB

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # Ignored
        space_held = action[1] == 1
        shift_held = action[2] == 1  # Ignored

        reward = 0.1  # Survival reward

        # --- Game Logic ---
        self._handle_input(space_held)
        self._update_player()
        collision = self._update_cacti()
        self._update_particles()
        self._spawn_cacti()
        self._update_difficulty()
        
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1
        if self.hit_flash_timer > 0:
            self.hit_flash_timer -= 1

        if collision:
            reward = -5.0
            self.lives -= 1
            # sfx: player_hit.wav
            self.invincibility_timer = self.FPS * 2 # 2 seconds of invincibility
            self.hit_flash_timer = self.FPS // 4 # 0.25 seconds of flash
            if self.lives <= 0:
                self.game_over = True
                # sfx: game_over.wav

        self.steps += 1
        self.score = self.steps # Score is distance traveled

        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over: # Survived full episode
            reward += 50.0
            # sfx: level_complete.wav

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, space_held):
        # Jump on the rising edge of the space press, only if on the ground
        if space_held and not self.space_was_held and not self.is_jumping:
            self.dino_vy = self.JUMP_STRENGTH
            self.is_jumping = True
            # sfx: jump.wav
        self.space_was_held = space_held

    def _update_player(self):
        if self.is_jumping:
            self.dino_vy += self.GRAVITY
            self.dino_y += self.dino_vy

        if self.dino_y >= self.GROUND_Y:
            if self.is_jumping: # Just landed
                # sfx: land.wav
                self._create_dust_particles(self.DINO_X + self.DINO_WIDTH / 2, self.GROUND_Y + self.DINO_HEIGHT)
            self.dino_y = self.GROUND_Y
            self.dino_vy = 0
            self.is_jumping = False

    def _update_cacti(self):
        collision_detected = False
        dino_hitbox = pygame.Rect(self.DINO_X + 5, self.dino_y + 5, self.DINO_WIDTH - 10, self.DINO_HEIGHT - 5)

        for cactus in self.cacti:
            cactus['x'] -= self.cactus_speed
            cactus_hitbox = pygame.Rect(cactus['x'], cactus['y'], cactus['width'], cactus['height'])
            
            if dino_hitbox.colliderect(cactus_hitbox) and self.invincibility_timer == 0:
                collision_detected = True

        self.cacti = [c for c in self.cacti if c['x'] + c['width'] > 0]
        return collision_detected

    def _spawn_cacti(self):
        # Don't spawn if the last cactus is too close
        if self.cacti and self.cacti[-1]['x'] > self.WIDTH - 200:
            return

        if self.np_random.random() < self.cactus_spawn_prob:
            cactus_type = self.np_random.integers(0, 3)
            if cactus_type == 0: # Single small
                height = 35
                width = 20
            elif cactus_type == 1: # Single tall
                height = 50
                width = 25
            else: # Double
                height = 35
                width = 50
            
            self.cacti.append({
                'x': self.WIDTH,
                'y': self.GROUND_Y + self.DINO_HEIGHT - height,
                'width': width,
                'height': height,
                'type': cactus_type
            })
            # sfx: cactus_spawn.wav

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.SPEED_INCREASE_INTERVAL == 0:
            self.cactus_speed += self.SPEED_INCREASE_AMOUNT
            self.cactus_spawn_prob += self.SPAWN_PROB_INCREASE_AMOUNT

    def _create_dust_particles(self, x, y):
        for _ in range(10):
            self.particles.append({
                'x': x + self.np_random.uniform(-10, 10),
                'y': y + self.np_random.uniform(-5, 5),
                'vx': self.np_random.uniform(-2, 0),
                'vy': self.np_random.uniform(-1, 0),
                'life': self.FPS // 2 # 0.5 second lifespan
            })

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_SKY)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        if self.hit_flash_timer > 0:
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            flash_surface.fill(self.COLOR_HIT_FLASH)
            self.screen.blit(flash_surface, (0, 0))

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y + self.DINO_HEIGHT, self.WIDTH, self.HEIGHT - (self.GROUND_Y + self.DINO_HEIGHT)))
        # Sun
        pygame.gfxdraw.filled_circle(self.screen, self.WIDTH - 60, 60, 30, (255, 255, 0))
        pygame.gfxdraw.aacircle(self.screen, self.WIDTH - 60, 60, 30, (255, 255, 0))

    def _render_game(self):
        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / (self.FPS / 2)))))
            color = (*self.COLOR_DUST, alpha)
            temp_surf = pygame.Surface((3, 3), pygame.SRCALPHA)
            temp_surf.fill(color)
            self.screen.blit(temp_surf, (int(p['x']), int(p['y'])))

        # Draw Cacti
        for cactus in self.cacti:
            pygame.draw.rect(self.screen, self.COLOR_CACTUS, (int(cactus['x']), int(cactus['y']), cactus['width'], cactus['height']))

        # Draw Dino
        dino_is_visible = self.invincibility_timer == 0 or (self.steps % 10 < 5)
        if dino_is_visible:
            self._draw_dino(int(self.DINO_X), int(self.dino_y))

    def _draw_dino(self, x, y):
        body = pygame.Rect(x + 10, y, 30, 30)
        head = pygame.Rect(x + 30, y, 10, 10)
        tail = pygame.Rect(x, y + 15, 10, 10)
        pygame.draw.rect(self.screen, self.COLOR_DINO, body)
        pygame.draw.rect(self.screen, self.COLOR_DINO, head)
        pygame.draw.rect(self.screen, self.COLOR_DINO, tail)
        
        # Eye
        pygame.draw.rect(self.screen, self.COLOR_SKY, (x + 33, y + 2, 3, 3))

        # Legs
        leg_y = y + 30
        if self.is_jumping:
            # Tucked legs
            pygame.draw.rect(self.screen, self.COLOR_DINO, (x + 15, leg_y, 8, 8))
            pygame.draw.rect(self.screen, self.COLOR_DINO, (x + 28, leg_y, 8, 8))
        else:
            # Running animation
            cycle = (self.steps // 4) % 2
            leg1_h = 14 if cycle == 0 else 10
            leg2_h = 10 if cycle == 0 else 14
            pygame.draw.rect(self.screen, self.COLOR_DINO, (x + 15, leg_y, 8, leg1_h))
            pygame.draw.rect(self.screen, self.COLOR_DINO, (x + 28, leg_y, 8, leg2_h))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score:05d}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives (Hearts)
        for i in range(self.lives):
            self._draw_heart(self.WIDTH - 30 - (i * 35), 25, self.COLOR_HEART)

    def _draw_heart(self, x, y, color):
        # A simple heart shape using polygons
        points = [
            (x, y - 5), (x + 5, y - 10), (x + 10, y - 5),
            (x, y + 5),
            (x - 10, y - 5), (x - 5, y - 10)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "cactus_speed": self.cactus_speed
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Dino Jumper")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not terminated:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        space_held = keys[pygame.K_SPACE]
        
        # Convert to MultiDiscrete action
        # actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right) -> not used
        # actions[1]: Space button (0=released, 1=held)
        # actions[2]: Shift button (0=released, 1=held) -> not used
        action = [0, 1 if space_held else 0, 0]

        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering for human play ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event handling and clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()