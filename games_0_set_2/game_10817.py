import gymnasium as gym
import os
import pygame
import numpy as np
import math
import random
from collections import deque
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:01:28.319015
# Source Brief: brief_00817.md
# Brief Index: 817
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Catch falling blocks with your paddle. Stack blocks of the same color to create chains and clear them for points. "
        "Reach the target score before time runs out!"
    )
    user_guide = "Use the ← and → arrow keys to move the paddle left and right to catch the falling blocks."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.UI_HEIGHT = 40
        self.PLAY_AREA_HEIGHT = self.HEIGHT - self.UI_HEIGHT
        self.FPS = 30
        self.GAME_DURATION_SECONDS = 60
        self.MAX_STEPS = self.GAME_DURATION_SECONDS * self.FPS

        # --- Entity Properties ---
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 16
        self.PADDLE_SPEED = 8
        self.BLOCK_SIZE = 32
        self.MINI_BLOCK_SIZE = 12
        self.BLOCK_FALL_SPEED = 120  # pixels per second
        self.MINI_BLOCK_FALL_SPEED = 240 # pixels per second
        self.GRID_COLS = self.WIDTH // self.BLOCK_SIZE
        self.GRID_ROWS = self.PLAY_AREA_HEIGHT // self.BLOCK_SIZE

        # --- Gameplay Parameters ---
        self.WIN_SCORE = 1000
        self.INITIAL_SPAWN_RATE = 1.0  # blocks per second
        self.MAX_SPAWN_RATE = 2.0
        self.DIFFICULTY_INTERVAL = 10 # seconds

        # --- Visuals ---
        self.COLORS = [
            (255, 70, 70),   # Red
            (70, 255, 70),   # Green
            (70, 130, 255),  # Blue
            (255, 220, 70)   # Yellow
        ]
        self.COLOR_BG = (10, 15, 25)
        self.COLOR_PADDLE = (220, 220, 255)
        self.COLOR_PADDLE_GLOW = (150, 150, 255)
        self.COLOR_UI_TEXT = (230, 230, 240)
        self.COLOR_BORDER = (40, 50, 70)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_huge = pygame.font.Font(None, 64)

        # --- Internal State (initialized in reset) ---
        self.paddle = None
        self.falling_blocks = None
        self.static_blocks = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.time_remaining = None
        self.block_spawn_timer = None
        self.block_spawn_rate = None
        self.background_stars = None
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = float(self.GAME_DURATION_SECONDS)

        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.PLAY_AREA_HEIGHT - self.PADDLE_HEIGHT,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self.falling_blocks = []
        self.static_blocks = {}  # Using dict for sparse grid: {(col, row): block_info}
        self.particles = []

        self.block_spawn_timer = 0
        self.block_spawn_rate = self.INITIAL_SPAWN_RATE

        if self.background_stars is None:
            self.background_stars = [
                (random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT), random.randint(1, 2))
                for _ in range(100)
            ]

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        dt = 1.0 / self.FPS

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_remaining -= dt

        # 1. Handle Player Input
        self._handle_input(action)

        # 2. Update Game Logic
        self._update_difficulty()
        self._spawn_falling_blocks(dt)

        newly_landed_coords = self._update_falling_blocks(dt)
        if newly_landed_coords:
            # Process matches and get reward
            match_reward = self._process_matches(newly_landed_coords)
            reward += match_reward

        self._update_particles(dt)

        # 3. Check for Termination
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward += 100  # Win bonus
            terminated = True
        elif self.time_remaining <= 0:
            reward -= 100  # Lose penalty
            terminated = True

        self.game_over = terminated
        truncated = self.steps >= self.MAX_STEPS

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement = action[0]
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED

        self.paddle.x = max(0, min(self.WIDTH - self.PADDLE_WIDTH, self.paddle.x))

    def _update_difficulty(self):
        time_elapsed = self.GAME_DURATION_SECONDS - self.time_remaining
        difficulty_level = int(time_elapsed // self.DIFFICULTY_INTERVAL)
        self.block_spawn_rate = min(
            self.MAX_SPAWN_RATE,
            self.INITIAL_SPAWN_RATE + difficulty_level * 0.25
        )

    def _spawn_falling_blocks(self, dt):
        self.block_spawn_timer += dt * self.block_spawn_rate
        if self.block_spawn_timer >= 1.0:
            self.block_spawn_timer -= 1.0
            spawn_col = self.np_random.integers(0, self.GRID_COLS)
            new_block = {
                "rect": pygame.Rect(spawn_col * self.BLOCK_SIZE, -self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE),
                "color": self.COLORS[self.np_random.integers(0, len(self.COLORS))],
                "is_mini": False
            }
            self.falling_blocks.append(new_block)

    def _update_falling_blocks(self, dt):
        newly_landed_coords = []
        for block in self.falling_blocks[:]:
            fall_speed = self.MINI_BLOCK_FALL_SPEED if block["is_mini"] else self.BLOCK_FALL_SPEED
            block["rect"].y += fall_speed * dt

            # Check for landing
            landed = False
            target_y = self.PLAY_AREA_HEIGHT - block["rect"].height
            
            # Collision with static blocks
            block_col = int(block["rect"].centerx // self.BLOCK_SIZE)
            for r in range(self.GRID_ROWS):
                if (block_col, r) in self.static_blocks:
                    target_y = min(target_y, self.static_blocks[(block_col, r)]["rect"].top - block["rect"].height)
            
            # Collision with paddle (only for non-mini blocks)
            if not block["is_mini"] and self.paddle.colliderect(block["rect"]):
                 target_y = min(target_y, self.paddle.top - block["rect"].height)

            if block["rect"].y >= target_y:
                landed = True
                block["rect"].y = target_y
            
            if landed:
                self.falling_blocks.remove(block)
                
                # Convert to static block and snap to grid
                land_col = int(block["rect"].centerx // self.BLOCK_SIZE)
                land_row = int(block["rect"].bottom // self.BLOCK_SIZE) -1
                
                # Prevent stacking outside bounds
                land_col = max(0, min(self.GRID_COLS - 1, land_col))
                land_row = max(0, min(self.GRID_ROWS - 1, land_row))

                if (land_col, land_row) not in self.static_blocks:
                    static_rect = pygame.Rect(land_col * self.BLOCK_SIZE, land_row * self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE)
                    self.static_blocks[(land_col, land_row)] = {"rect": static_rect, "color": block["color"]}
                    newly_landed_coords.append((land_col, land_row))

        return newly_landed_coords

    def _process_matches(self, newly_landed_coords):
        total_reward = 0
        checked_coords = set()
        successful_matches = []

        for coord in newly_landed_coords:
            if coord in checked_coords or coord not in self.static_blocks:
                continue
            
            match_group = self._find_matches(coord)
            if len(match_group) >= 3:
                successful_matches.append(match_group)
                checked_coords.update(match_group)
        
        for match_group in successful_matches:
            total_reward += 1.0  # +1 for the chain
            if len(match_group) >= 6:
                total_reward += 5.0  # Bonus for big chain

            for block_coord in match_group:
                if block_coord in self.static_blocks:
                    cleared_block = self.static_blocks.pop(block_coord)
                    total_reward += 0.1
                    self.score += 10

                    self._spawn_particles(cleared_block["rect"].center, cleared_block["color"], 15)

                    # Spawn a new falling mini-block
                    mini_block = {
                        "rect": pygame.Rect(cleared_block["rect"].x + (self.BLOCK_SIZE - self.MINI_BLOCK_SIZE) / 2,
                                           cleared_block["rect"].y + (self.BLOCK_SIZE - self.MINI_BLOCK_SIZE) / 2,
                                           self.MINI_BLOCK_SIZE, self.MINI_BLOCK_SIZE),
                        "color": cleared_block["color"],
                        "is_mini": True
                    }
                    self.falling_blocks.append(mini_block)
        return total_reward

    def _find_matches(self, start_coord):
        if start_coord not in self.static_blocks:
            return []
        
        target_color = self.static_blocks[start_coord]["color"]
        q = deque([start_coord])
        visited = {start_coord}
        match_group = []

        while q:
            col, row = q.popleft()
            match_group.append((col, row))
            
            for dc, dr in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_coord = (col + dc, row + dr)
                if next_coord not in visited and next_coord in self.static_blocks:
                    if self.static_blocks[next_coord]["color"] == target_color:
                        visited.add(next_coord)
                        q.append(next_coord)
        return match_group

    def _spawn_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(20, 100)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "radius": self.np_random.uniform(2, 6),
                "color": color,
                "life": self.np_random.uniform(0.5, 1.5)
            })

    def _update_particles(self, dt):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0] * dt
            p["pos"][1] += p["vel"][1] * dt
            p["vel"][1] += 100 * dt # Gravity on particles
            p["life"] -= dt
            if p["life"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background stars
        for x, y, r in self.background_stars:
            pygame.draw.circle(self.screen, (50, 60, 80), (x, y), r)
        
        # Static blocks
        for block in self.static_blocks.values():
            pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=4)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in block["color"]), block["rect"], width=2, border_radius=4)

        # Falling blocks
        for block in self.falling_blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=4 if not block["is_mini"] else 2)

        # Paddle glow
        glow_rect = self.paddle.inflate(10, 10)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.COLOR_PADDLE_GLOW, 60), glow_surf.get_rect(), border_radius=12)
        self.screen.blit(glow_surf, glow_rect.topleft)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=8)
        
        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / 1.0))))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(temp_surf, (int(p["pos"][0] - p["radius"]), int(p["pos"][1] - p["radius"])), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # UI Background
        pygame.draw.rect(self.screen, self.COLOR_BORDER, (0, self.PLAY_AREA_HEIGHT, self.WIDTH, self.UI_HEIGHT))
        pygame.draw.line(self.screen, self.COLOR_PADDLE, (0, self.PLAY_AREA_HEIGHT), (self.WIDTH, self.PLAY_AREA_HEIGHT), 2)
        
        # Score Text
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, self.HEIGHT - self.UI_HEIGHT + 5))
        
        # Time Text
        time_str = f"TIME: {math.ceil(max(0, self.time_remaining))}"
        time_text = self.font_large.render(time_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 15, self.HEIGHT - self.UI_HEIGHT + 5))

        # Game Over / Win Text
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.WIN_SCORE:
                end_text = self.font_huge.render("YOU WIN!", True, (150, 255, 150))
            else:
                end_text = self.font_huge.render("GAME OVER", True, (255, 100, 100))
            
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # It will not be executed by the autograder
    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    # Un-dummy the video driver for local play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Colorfall Chains")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action[0] = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action[0] = 4
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before auto-resetting, or wait for R key
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)
        
    env.close()
    pygame.quit()