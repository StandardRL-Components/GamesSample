import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:54:31.320976
# Source Brief: brief_02455.md
# Brief Index: 2455
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a shrinking square.
    The goal is to collect 10 larger green squares to win.
    The player's square shrinks with every step, and collecting a green
    square resets its size. The episode ends if the player's square
    disappears, if all green squares are collected, or if the step
    limit is reached.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a shrinking square and collect all the green targets to win. "
        "Collecting a target resets your size, but you shrink with every passing moment!"
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move your square. "
        "Collect all green targets before you shrink away."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PLAYER_START_SIZE = 20.0
    PLAYER_SPEED = 5.0
    SHRINK_RATE = 0.99  # 1% size reduction per step
    COLLECTIBLE_SIZE = 30
    NUM_COLLECTIBLES = 10
    MAX_STEPS = 2000

    # --- COLORS (Bright and Contrasting) ---
    COLOR_BG = (10, 15, 25)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 75, 128)
    COLOR_COLLECTIBLE = (100, 255, 100)
    COLOR_COLLECTIBLE_GLOW = (50, 128, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_UI_BAR_BG = (40, 40, 60)
    COLOR_UI_BAR_FILL = (0, 150, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 20, bold=True)

        # --- State Variables ---
        self.steps = None
        self.score = None
        self.game_over = None
        self.player_pos = None
        self.player_size = None
        self.collectibles = None
        self.particles = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Initialize Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = np.array([self.SCREEN_WIDTH / 2.0, self.SCREEN_HEIGHT / 2.0])
        self.player_size = self.PLAYER_START_SIZE
        
        self.particles = []
        self._spawn_collectibles()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # --- Unpack Action ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Update Game Logic ---
        self.steps += 1
        self._handle_movement(movement)
        self.player_size *= self.SHRINK_RATE
        self._update_particles()
        
        reward = self._handle_collisions()

        # --- Check Termination Conditions ---
        terminated = False
        if self.player_size < 1.0:
            reward = -100.0  # Loss penalty
            terminated = True
        elif self.score >= self.NUM_COLLECTIBLES:
            reward = 100.0   # Win bonus
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True # Max steps reached

        if not terminated:
            reward += 0.1 # Survival reward
            
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _spawn_collectibles(self):
        self.collectibles = []
        placed_rects = []
        margin = 10
        player_start_area = pygame.Rect(
            self.player_pos[0] - 50, self.player_pos[1] - 50, 100, 100
        )

        for _ in range(self.NUM_COLLECTIBLES):
            for _ in range(100): # Max 100 placement attempts
                pos = np.array([
                    self.np_random.uniform(margin, self.SCREEN_WIDTH - self.COLLECTIBLE_SIZE - margin),
                    self.np_random.uniform(margin, self.SCREEN_HEIGHT - self.COLLECTIBLE_SIZE - margin)
                ])
                new_rect = pygame.Rect(pos[0], pos[1], self.COLLECTIBLE_SIZE, self.COLLECTIBLE_SIZE)

                # Check for overlap with other collectibles and player start area
                if new_rect.collidelist(placed_rects) == -1 and not new_rect.colliderect(player_start_area):
                    self.collectibles.append(pos)
                    placed_rects.append(new_rect)
                    break

    def _handle_movement(self, movement):
        if movement == 1: # Up
            self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2: # Down
            self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3: # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_pos[0] += self.PLAYER_SPEED

        # Clamp player position to screen bounds
        half_size = self.player_size / 2
        self.player_pos[0] = np.clip(self.player_pos[0], half_size, self.SCREEN_WIDTH - half_size)
        self.player_pos[1] = np.clip(self.player_pos[1], half_size, self.SCREEN_HEIGHT - half_size)

    def _handle_collisions(self):
        reward = 0.0
        player_rect = pygame.Rect(
            self.player_pos[0] - self.player_size / 2,
            self.player_pos[1] - self.player_size / 2,
            self.player_size,
            self.player_size
        )

        for i in range(len(self.collectibles) - 1, -1, -1):
            collectible_pos = self.collectibles[i]
            collectible_rect = pygame.Rect(
                collectible_pos[0], collectible_pos[1], self.COLLECTIBLE_SIZE, self.COLLECTIBLE_SIZE
            )

            if player_rect.colliderect(collectible_rect):
                self.score += 1
                reward += 10.0
                self.player_size = self.PLAYER_START_SIZE
                self._spawn_particles(collectible_pos + self.COLLECTIBLE_SIZE / 2)
                self.collectibles.pop(i)
                break
        return reward

    def _spawn_particles(self, pos):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            velocity = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            lifespan = self.np_random.integers(20, 40)
            self.particles.append({
                'pos': pos.copy(),
                'vel': velocity,
                'life': lifespan,
                'max_life': lifespan,
                'color': random.choice([self.COLOR_COLLECTIBLE, self.COLOR_PLAYER, (200, 200, 255)])
            })

    def _update_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # friction
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.pop(i)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render Collectibles with pulsating glow
        pulse = (math.sin(self.steps * 0.1) + 1) / 2  # 0 to 1
        glow_size_add = 5 + pulse * 10
        for pos in self.collectibles:
            # Glow
            glow_rect = pygame.Rect(
                pos[0] - glow_size_add / 2,
                pos[1] - glow_size_add / 2,
                self.COLLECTIBLE_SIZE + glow_size_add,
                self.COLLECTIBLE_SIZE + glow_size_add
            )
            pygame.draw.rect(self.screen, self.COLOR_COLLECTIBLE_GLOW, glow_rect, border_radius=8)
            # Solid
            solid_rect = pygame.Rect(pos[0], pos[1], self.COLLECTIBLE_SIZE, self.COLLECTIBLE_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_COLLECTIBLE, solid_rect, border_radius=6)

        # Render Player with glow
        player_size_int = max(1, int(self.player_size))
        player_rect = pygame.Rect(
            int(self.player_pos[0] - player_size_int / 2),
            int(self.player_pos[1] - player_size_int / 2),
            player_size_int,
            player_size_int
        )
        # Glow
        glow_rect = player_rect.inflate(player_size_int * 0.8, player_size_int * 0.8)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_GLOW, glow_rect, border_radius=int(player_size_int * 0.4))
        # Solid
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=int(player_size_int * 0.2))

        # Render Particles
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            radius = int(life_ratio * 5)
            if radius > 0:
                pos_int = (int(p['pos'][0]), int(p['pos'][1]))
                alpha = int(life_ratio * 255)
                color = p['color']
                # Using gfxdraw for anti-aliased circles
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius, (*color, alpha))
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, (*color, alpha))

    def _render_ui(self):
        # Score Text
        score_text = self.font.render(f"COLLECTED: {self.score}/{self.NUM_COLLECTIBLES}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Player Size Bar
        bar_width = 200
        bar_height = 15
        bar_x = (self.SCREEN_WIDTH - bar_width) / 2
        bar_y = self.SCREEN_HEIGHT - bar_height - 10

        size_ratio = max(0, self.player_size / self.PLAYER_START_SIZE)
        fill_width = int(bar_width * size_ratio)

        # Background of the bar
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        # Fill of the bar
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_UI_BAR_FILL, (bar_x, bar_y, fill_width, bar_height), border_radius=4)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_size": self.player_size,
            "collectibles_left": len(self.collectibles)
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    done = False
    total_reward = 0
    
    # --- Pygame setup for manual play ---
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Shrinking Square Collector")
    clock = pygame.time.Clock()

    print("\n--- Manual Control ---")
    print("Arrows: Move")
    print("R: Reset")
    print("Q: Quit")
    
    while not done:
        # --- Action Mapping for Human ---
        movement = 0 # no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print(f"--- Game Reset ---")

        # --- Environment Step ---
        if not env.game_over:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated:
                print(f"Episode Finished! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
        
        # --- Rendering ---
        # Convert the observation (H, W, C) to a Pygame surface (W, H)
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS for smooth play

    env.close()