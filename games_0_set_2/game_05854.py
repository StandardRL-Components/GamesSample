
# Generated: 2025-08-28T06:17:33.363255
# Source Brief: brief_05854.md
# Brief Index: 5854

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from typing import List, Tuple, Set, Dict, Any
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper classes for game entities
class Scarecrow:
    def __init__(self, path: List[Tuple[int, int]], speed: int, start_index: int = 0):
        self.path = path
        self.speed = speed  # Moves every 'speed' frames
        self.path_index = start_index % len(self.path)
        self.pos = self.path[self.path_index]
        self.move_counter = 0
        self.direction = 1  # 1 for forward, -1 for backward

    def update(self):
        self.move_counter += 1
        if self.move_counter >= self.speed:
            self.move_counter = 0
            
            # Simple back-and-forth patrol logic
            if not (0 <= self.path_index + self.direction < len(self.path)):
                self.direction *= -1
            
            self.path_index += self.direction
            self.pos = self.path[self.path_index]

    def reset(self, start_index: int):
        self.path_index = start_index % len(self.path)
        self.pos = self.path[self.path_index]
        self.move_counter = 0
        self.direction = 1

class Particle:
    def __init__(self, x, y, color, max_life=20):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        self.x = x
        self.y = y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = max_life
        self.max_life = max_life
        self.color = color

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            radius = int(max(0, (self.life / self.max_life) * 5))
            alpha = int(max(0, (self.life / self.max_life) * 255))
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*self.color, alpha), (radius, radius), radius)
            surface.blit(temp_surf, (int(self.x - radius), int(self.y - radius)))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your character (white dot). "
        "Space and Shift do not have a function in this game."
    )

    game_description = (
        "Evade patrolling scarecrows and harvest 20 glowing crops in a moonlit field before the 90-second timer runs out. "
        "Getting caught by a scarecrow ends the game immediately."
    )

    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 20, 12
    CELL_SIZE = 32
    GAME_AREA_HEIGHT = GRID_HEIGHT * CELL_SIZE # 384
    UI_AREA_Y_START = GAME_AREA_HEIGHT
    
    FPS = 30
    TIME_LIMIT_SECONDS = 90
    WIN_SCORE = 20
    INITIAL_CROPS = 30

    # --- Colors ---
    COLOR_BG = (10, 20, 30)
    COLOR_FENCE = (40, 30, 20)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_CROP_MAIN = (100, 200, 255)
    COLOR_CROP_GLOW = (50, 100, 200)
    COLOR_SCARECROW = (25, 15, 5)
    COLOR_SCARECROW_EYES = (255, 20, 20)
    COLOR_UI_TEXT = (220, 220, 200)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        try:
            self.font_main = pygame.font.SysFont("Consolas", 24)
            self.font_title = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_main = pygame.font.SysFont(None, 28)
            self.font_title = pygame.font.SysFont(None, 56)

        # Game state variables initialized in reset()
        self.steps = 0
        self.score = 0
        self.timer = 0
        self.game_over = False
        self.win = False
        
        self.player_grid_pos = (0, 0)
        self.player_pixel_pos = np.array([0.0, 0.0])
        self.crops: Set[Tuple[int, int]] = set()
        self.scarecrows: List[Scarecrow] = []
        self.particles: List[Particle] = []
        
        self.prev_dist_to_closest_crop = 0
        self.prev_dist_to_closest_scarecrow = 0
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.timer = self.FPS * self.TIME_LIMIT_SECONDS
        self.game_over = False
        self.win = False
        self.particles.clear()
        
        # --- Player ---
        self.player_grid_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT - 1)
        self.player_pixel_pos = self._grid_to_pixel(self.player_grid_pos)

        # --- Crops ---
        self.crops.clear()
        possible_positions = set((x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT))
        possible_positions.discard(self.player_grid_pos)
        
        num_crops = min(self.INITIAL_CROPS, len(possible_positions))
        crop_indices = self.np_random.choice(len(possible_positions), num_crops, replace=False)
        crop_list = list(possible_positions)
        self.crops = {crop_list[i] for i in crop_indices}

        # --- Scarecrows ---
        self._initialize_scarecrows()

        # --- Reward state ---
        self.prev_dist_to_closest_crop = self._get_dist_to_closest(self.player_grid_pos, self.crops)
        scarecrow_positions = {s.pos for s in self.scarecrows}
        self.prev_dist_to_closest_scarecrow = self._get_dist_to_closest(self.player_grid_pos, scarecrow_positions)

        return self._get_observation(), self._get_info()
    
    def _initialize_scarecrows(self):
        self.scarecrows.clear()
        # Horizontal patrol
        path1 = [(x, 2) for x in range(1, self.GRID_WIDTH - 1)]
        self.scarecrows.append(Scarecrow(path=path1, speed=5, start_index=0))
        # Vertical patrol
        path2 = [(self.GRID_WIDTH - 4, y) for y in range(1, self.GRID_HEIGHT - 1)]
        self.scarecrows.append(Scarecrow(path=path2, speed=4, start_index=self.np_random.integers(len(path2))))
        # Box patrol
        path3 = (
            [(x, 5) for x in range(3, 8)] + 
            [(7, y) for y in range(6, 9)] + 
            [(x, 8) for x in range(6, 2, -1)] +
            [(3, y) for y in range(7, 4, -1)]
        )
        self.scarecrows.append(Scarecrow(path=path3, speed=6, start_index=self.np_random.integers(len(path3))))
        # L-shape patrol
        path4 = (
            [(x, self.GRID_HEIGHT - 2) for x in range(12, 18)] +
            [(12, y) for y in range(self.GRID_HEIGHT - 3, 3, -1)]
        )
        self.scarecrows.append(Scarecrow(path=path4, speed=5, start_index=self.np_random.integers(len(path4))))

        for s in self.scarecrows:
            # Ensure no scarecrow starts on a crop
            if s.pos in self.crops:
                self.crops.remove(s.pos)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self.steps += 1
        self.timer -= 1
        reward = 0
        
        # --- Update Player Position ---
        px, py = self.player_grid_pos
        if movement == 1 and py > 0: py -= 1
        elif movement == 2 and py < self.GRID_HEIGHT - 1: py += 1
        elif movement == 3 and px > 0: px -= 1
        elif movement == 4 and px < self.GRID_WIDTH - 1: px += 1
        
        # Only update if there was a valid move
        if (px, py) != self.player_grid_pos:
            self.player_grid_pos = (px, py)

        # --- Update Scarecrows ---
        for s in self.scarecrows:
            s.update()

        # --- Update Particles ---
        for p in self.particles[:]:
            p.update()
            if p.life <= 0:
                self.particles.remove(p)

        # --- Continuous Reward Calculation ---
        scarecrow_positions = {s.pos for s in self.scarecrows}
        dist_crop = self._get_dist_to_closest(self.player_grid_pos, self.crops)
        dist_scarecrow = self._get_dist_to_closest(self.player_grid_pos, scarecrow_positions)

        # Reward for getting closer to a crop
        reward += 1.0 * (self.prev_dist_to_closest_crop - dist_crop)
        # Penalty for getting closer to a scarecrow
        reward -= 0.5 * (self.prev_dist_to_closest_scarecrow - dist_scarecrow)

        self.prev_dist_to_closest_crop = dist_crop
        self.prev_dist_to_closest_scarecrow = dist_scarecrow
        
        # --- Collision and Collection ---
        if self.player_grid_pos in self.crops:
            self.crops.remove(self.player_grid_pos)
            self.score += 1
            reward += 10  # Event reward for collecting
            # SFX: Crop collect sound
            px, py = self._grid_to_pixel(self.player_grid_pos)
            for _ in range(20):
                self.particles.append(Particle(px, py, self.COLOR_CROP_MAIN))

        if self.player_grid_pos in scarecrow_positions:
            self.game_over = True
            reward -= 100  # Terminal penalty for being caught
            # SFX: Player caught sound
        
        # --- Termination Check ---
        terminated = False
        if self.score >= self.WIN_SCORE:
            self.game_over = True
            self.win = True
            terminated = True
            reward += 100  # Terminal reward for winning
        elif self.timer <= 0:
            self.game_over = True
            terminated = True
            reward -= 50  # Terminal penalty for timeout
        elif self.game_over: # Caught by scarecrow
            terminated = True

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _grid_to_pixel(self, grid_pos):
        x, y = grid_pos
        return np.array([
            x * self.CELL_SIZE + self.CELL_SIZE / 2,
            y * self.CELL_SIZE + self.CELL_SIZE / 2
        ])

    def _get_dist_to_closest(self, pos: Tuple[int, int], targets: Set[Tuple[int, int]]) -> float:
        if not targets:
            return 0
        px, py = pos
        min_dist = float('inf')
        for tx, ty in targets:
            dist = abs(px - tx) + abs(py - ty) # Manhattan distance
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _get_observation(self):
        # Interpolate player visual position for smooth movement
        target_pixel_pos = self._grid_to_pixel(self.player_grid_pos)
        self.player_pixel_pos = self.player_pixel_pos * 0.5 + target_pixel_pos * 0.5

        # --- Drawing ---
        self.screen.fill(self.COLOR_BG)
        
        # Draw fence
        fence_rect = pygame.Rect(0, 0, self.GRID_WIDTH * self.CELL_SIZE, self.GAME_AREA_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_FENCE, fence_rect, 4)

        # Draw crops
        pulse = (math.sin(self.steps * 0.1) + 1) / 2  # Varies between 0 and 1
        glow_radius = int(self.CELL_SIZE * 0.5 + pulse * 4)
        for x, y in self.crops:
            px = x * self.CELL_SIZE + self.CELL_SIZE // 2
            py = y * self.CELL_SIZE + self.CELL_SIZE // 2
            pygame.gfxdraw.filled_circle(self.screen, px, py, glow_radius, (*self.COLOR_CROP_GLOW, 100))
            pygame.gfxdraw.filled_circle(self.screen, px, py, self.CELL_SIZE // 4, self.COLOR_CROP_MAIN)
            pygame.gfxdraw.aacircle(self.screen, px, py, self.CELL_SIZE // 4, self.COLOR_CROP_MAIN)

        # Draw scarecrows
        for s in self.scarecrows:
            sx, sy = self._grid_to_pixel(s.pos)
            scare_rect = pygame.Rect(sx - self.CELL_SIZE//3, sy - self.CELL_SIZE//2, self.CELL_SIZE*2//3, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_SCARECROW, scare_rect)
            # Eyes
            eye_y = sy - self.CELL_SIZE // 4
            eye_glow_radius = int(2 + pulse * 2)
            pygame.draw.circle(self.screen, self.COLOR_SCARECROW_EYES, (sx - 5, eye_y), 2)
            pygame.draw.circle(self.screen, self.COLOR_SCARECROW_EYES, (sx + 5, eye_y), 2)
            pygame.gfxdraw.filled_circle(self.screen, int(sx-5), int(eye_y), eye_glow_radius, (*self.COLOR_SCARECROW_EYES, 150))
            pygame.gfxdraw.filled_circle(self.screen, int(sx+5), int(eye_y), eye_glow_radius, (*self.COLOR_SCARECROW_EYES, 150))

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)
        
        # Draw player
        ppx, ppy = int(self.player_pixel_pos[0]), int(self.player_pixel_pos[1])
        pygame.gfxdraw.filled_circle(self.screen, ppx, ppy, self.CELL_SIZE // 3, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, ppx, ppy, self.CELL_SIZE // 3, self.COLOR_PLAYER)

        # Draw UI
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_ui(self):
        # UI Background
        ui_rect = pygame.Rect(0, self.UI_AREA_Y_START, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.UI_AREA_Y_START)
        pygame.draw.rect(self.screen, (0,0,0), ui_rect)

        # Score
        score_text = self.font_main.render(f"CROPS: {self.score} / {self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, self.UI_AREA_Y_START + 2))

        # Timer
        time_left = max(0, self.timer / self.FPS)
        timer_color = self.COLOR_SCARECROW_EYES if time_left < 10 else self.COLOR_UI_TEXT
        timer_text = self.font_main.render(f"TIME: {time_left:.1f}", True, timer_color)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, self.UI_AREA_Y_START + 2))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            if self.win:
                msg = "YOU SURVIVED"
                color = self.COLOR_CROP_MAIN
            else:
                msg = "CAUGHT!"
                color = self.COLOR_SCARECROW_EYES
            
            title_surf = self.font_title.render(msg, True, color)
            title_rect = title_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            
            overlay.blit(title_surf, title_rect)
            self.screen.blit(overlay, (0,0))
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": max(0, self.timer / self.FPS),
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Harvest Fright")
    
    running = True
    total_reward = 0
    
    # Map pygame keys to actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    while running:
        action = np.array([0, 0, 0]) # Default action is no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Game Reset ---")

        keys = pygame.key.get_pressed()
        
        # Movement (only one direction at a time)
        # Priority: Up > Down > Left > Right
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # Space and Shift
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait a bit before allowing reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            print("--- Game Reset ---")

        env.clock.tick(env.FPS)
        
    env.close()