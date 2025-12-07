import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T20:07:04.186992
# Source Brief: brief_03253.md
# Brief Index: 3253
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for a minimalist arcade game.

    The player controls a blue square that constantly shrinks. They must navigate it
    through a pair of oscillating red walls to reach a target at the top-center of
    the screen. Collecting green orbs allows the player to temporarily grow the
    square, giving them more time.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `action[0]` (Movement): 0=None, 1=Up, 2=Down, 3=Left, 4=Right
    - `action[1]` (Space): 1 to activate orb collection
    - `action[2]` (Shift): No effect

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    **Rewards:**
    - +100 for reaching the center target.
    - -100 for the square shrinking to its minimum size.
    - +5 for collecting a green orb.
    - Small positive/negative reward for moving up/down.

    **Termination:**
    - Reaching the target.
    - Square shrinking to minimum size.
    - Reaching the maximum step limit (1000).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a shrinking square, navigate through oscillating walls, and collect orbs to grow bigger as you race to the target at the top of the screen."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to collect green orbs and grow larger."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_PLAYER = (0, 200, 255)
    COLOR_PLAYER_GLOW = (0, 200, 255, 50)
    COLOR_WALL = (255, 50, 50)
    COLOR_WALL_GLOW = (255, 50, 50, 60)
    COLOR_ORB = (50, 255, 50)
    COLOR_ORB_GLOW = (50, 255, 50, 80)
    COLOR_TARGET = (255, 255, 255)
    COLOR_TARGET_GLOW = (255, 255, 255, 60)
    COLOR_UI = (220, 220, 220)
    COLOR_PARTICLE = (200, 255, 200)

    # Player
    INITIAL_PLAYER_SIZE = 30.0
    MINIMUM_PLAYER_SIZE = INITIAL_PLAYER_SIZE * 0.5
    PLAYER_SPEED = 4.0
    SHRINK_RATE = (INITIAL_PLAYER_SIZE * 0.01) / FPS # Shrink 1% of initial size per second

    # Walls
    WALL_BASE_FREQ = (2 * math.pi) / (5 * FPS) # 5-second period
    WALL_AMPLITUDE = 100
    WALL_CENTER_GAP = 120
    WALL_DIFFICULTY_INTERVAL = 500
    WALL_DIFFICULTY_INCREASE = 0.05

    # Orbs
    ORB_SIZE = 12
    ORB_SPAWN_WINDOW = (10 * FPS, 20 * FPS) # 10-20 seconds
    ORB_ENLARGE_AMOUNT = INITIAL_PLAYER_SIZE * 0.4

    # Target
    TARGET_POS = (SCREEN_WIDTH // 2, 50)
    TARGET_RADIUS = 15

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        
        self.player_pos = pygame.math.Vector2(0, 0)
        self.player_size = 0.0
        self.steps = 0
        self.score = 0.0
        self.wall_time = 0
        self.wall_speed_factor = 1.0
        self.orb = None
        self.orb_spawn_timer = 0
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        
        self.player_pos = pygame.math.Vector2(
            self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50
        )
        self.player_size = self.INITIAL_PLAYER_SIZE
        
        self.wall_time = 0
        self.wall_speed_factor = 1.0
        
        self.orb = None
        self.orb_spawn_timer = self.np_random.integers(
            self.ORB_SPAWN_WINDOW[0], self.ORB_SPAWN_WINDOW[1]
        )
        
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        space_held = action[1] == 1
        
        reward = 0.0
        terminated = False
        truncated = False
        
        # --- Update Game Logic ---
        self.steps += 1
        self.wall_time += 1

        # 1. Handle Player Movement
        prev_player_y = self.player_pos.y
        vel = pygame.math.Vector2(0, 0)
        if movement == 1: vel.y -= self.PLAYER_SPEED  # Up
        elif movement == 2: vel.y += self.PLAYER_SPEED  # Down
        elif movement == 3: vel.x -= self.PLAYER_SPEED  # Left
        elif movement == 4: vel.x += self.PLAYER_SPEED  # Right
        
        self.player_pos += vel
        
        # Clamp player position to screen bounds
        self.player_pos.x = np.clip(self.player_pos.x, self.player_size / 2, self.SCREEN_WIDTH - self.player_size / 2)
        self.player_pos.y = np.clip(self.player_pos.y, self.player_size / 2, self.SCREEN_HEIGHT - self.player_size / 2)

        # 2. Player Shrinking
        self.player_size -= self.SHRINK_RATE
        self.player_size = max(self.player_size, 0)

        # 3. Handle Orb Spawning
        if self.orb is None:
            self.orb_spawn_timer -= 1
            if self.orb_spawn_timer <= 0:
                self._spawn_orb()
        
        # 4. Handle Orb Collection
        if space_held and self.orb is not None:
            player_rect = self._get_player_rect()
            if player_rect.colliderect(self.orb['rect']):
                # Sound effect: Powerup
                reward += 5.0
                self.player_size += self.ORB_ENLARGE_AMOUNT
                self.player_size = min(self.player_size, self.INITIAL_PLAYER_SIZE * 1.5)
                self._create_particles(self.orb['pos'], 30, self.COLOR_PARTICLE)
                self.orb = None
                self.orb_spawn_timer = self.np_random.integers(
                    self.ORB_SPAWN_WINDOW[0], self.ORB_SPAWN_WINDOW[1]
                )

        # 5. Update Wall Difficulty
        if self.steps > 0 and self.steps % self.WALL_DIFFICULTY_INTERVAL == 0:
            self.wall_speed_factor += self.WALL_DIFFICULTY_INCREASE

        # 6. Update Particles
        self._update_particles()
        
        # --- Calculate Reward ---
        # Reward for vertical progress
        reward += (prev_player_y - self.player_pos.y) * 0.1

        # --- Check Termination Conditions ---
        # Victory: Reached target
        dist_to_target = self.player_pos.distance_to(self.TARGET_POS)
        if dist_to_target < self.TARGET_RADIUS + self.player_size / 2:
            terminated = True
            reward += 100.0
            # Sound effect: Victory
            self._create_particles(self.player_pos, 50, self.COLOR_TARGET)
        
        # Failure: Shrunk too much
        if self.player_size <= self.MINIMUM_PLAYER_SIZE:
            terminated = True
            reward -= 100.0
            # Sound effect: Failure
            self._create_particles(self.player_pos, 50, self.COLOR_WALL)

        # Timeout
        if self.steps >= self.MAX_STEPS:
            terminated = True
            truncated = True # Use truncated for timeout

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

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
            "player_size": self.player_size,
        }

    def _get_player_rect(self):
        return pygame.Rect(
            self.player_pos.x - self.player_size / 2,
            self.player_pos.y - self.player_size / 2,
            self.player_size,
            self.player_size
        )

    def _spawn_orb(self):
        pos_x = self.np_random.uniform(self.ORB_SIZE, self.SCREEN_WIDTH - self.ORB_SIZE)
        pos_y = self.np_random.uniform(100, self.SCREEN_HEIGHT - 100)
        pos = pygame.math.Vector2(pos_x, pos_y)
        rect = pygame.Rect(pos.x - self.ORB_SIZE, pos.y - self.ORB_SIZE, self.ORB_SIZE*2, self.ORB_SIZE*2)
        self.orb = {'pos': pos, 'rect': rect}

    def _create_particles(self, position, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': position.copy(), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['vel'] *= 0.95 # friction

    def _render_game(self):
        # Target
        pygame.gfxdraw.aacircle(self.screen, int(self.TARGET_POS[0]), int(self.TARGET_POS[1]), self.TARGET_RADIUS + 5, self.COLOR_TARGET_GLOW)
        pygame.gfxdraw.aacircle(self.screen, int(self.TARGET_POS[0]), int(self.TARGET_POS[1]), self.TARGET_RADIUS, self.COLOR_TARGET)
        pygame.gfxdraw.filled_circle(self.screen, int(self.TARGET_POS[0]), int(self.TARGET_POS[1]), self.TARGET_RADIUS, self.COLOR_TARGET_GLOW)

        # Walls
        offset = self.WALL_AMPLITUDE * math.sin(self.wall_time * self.WALL_BASE_FREQ * self.wall_speed_factor)
        left_wall_x = self.SCREEN_WIDTH / 2 - self.WALL_CENTER_GAP / 2 - offset
        right_wall_x = self.SCREEN_WIDTH / 2 + self.WALL_CENTER_GAP / 2 + offset
        
        pygame.draw.line(self.screen, self.COLOR_WALL_GLOW, (int(left_wall_x), 0), (int(left_wall_x), self.SCREEN_HEIGHT), 15)
        pygame.draw.line(self.screen, self.COLOR_WALL, (int(left_wall_x), 0), (int(left_wall_x), self.SCREEN_HEIGHT), 5)
        
        pygame.draw.line(self.screen, self.COLOR_WALL_GLOW, (int(right_wall_x), 0), (int(right_wall_x), self.SCREEN_HEIGHT), 15)
        pygame.draw.line(self.screen, self.COLOR_WALL, (int(right_wall_x), 0), (int(right_wall_x), self.SCREEN_HEIGHT), 5)

        # Orb
        if self.orb is not None:
            pos = self.orb['pos']
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), self.ORB_SIZE, self.COLOR_ORB_GLOW)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), self.ORB_SIZE, self.COLOR_ORB)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), self.ORB_SIZE - 2, self.COLOR_ORB)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30))
            color = (*p['color'], alpha)
            size = max(1, int(p['lifespan'] / 6))
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            self.screen.blit(temp_surf, (int(p['pos'].x - size), int(p['pos'].y - size)), special_flags=pygame.BLEND_RGBA_ADD)

        # Player
        player_rect = self._get_player_rect()
        glow_size = int(self.player_size * 1.5)
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PLAYER_GLOW, glow_surf.get_rect(), border_radius=max(1, glow_size//4))
        self.screen.blit(glow_surf, (player_rect.centerx - glow_size//2, player_rect.centery - glow_size//2), special_flags=pygame.BLEND_RGBA_ADD)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=max(1, int(self.player_size//4)))

    def _render_ui(self):
        # Score
        score_text = self.font.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))
        
        # Health/Size bar
        health_pct = np.clip((self.player_size - self.MINIMUM_PLAYER_SIZE) / (self.INITIAL_PLAYER_SIZE - self.MINIMUM_PLAYER_SIZE), 0, 1)
        bar_width = 200
        bar_height = 15
        bar_x = self.SCREEN_WIDTH - bar_width - 10
        bar_y = 10
        
        fill_width = int(bar_width * health_pct)
        fill_color = self.COLOR_PLAYER if health_pct > 0.25 else self.COLOR_WALL
        
        pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height), border_radius=3)
        if fill_width > 0:
            pygame.draw.rect(self.screen, fill_color, (bar_x, bar_y, fill_width, bar_height), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_UI, (bar_x, bar_y, bar_width, bar_height), 2, border_radius=3)

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game manually
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Minimalist Arcade")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0.0

    while not done:
        # --- Human Input ---
        movement = 0 # No-op
        space_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        
        action = [movement, space_held, 0] # Shift is not used

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0.0
            done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered image, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        clock.tick(env.FPS)

    env.close()