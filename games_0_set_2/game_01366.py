
# Generated: 2025-08-27T16:54:58.938280
# Source Brief: brief_01366.md
# Brief Index: 1366

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move one square at a time. Collect all souls to win. Avoid the specters!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a spooky graveyard, collecting souls while evading patrolling specters. Each move is a turn."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_W, SCREEN_H = 640, 400
    GRID_W, GRID_H = 16, 16
    CELL_SIZE = SCREEN_H // GRID_H  # 25
    GRID_PIXEL_W = GRID_W * CELL_SIZE # 400
    GRID_PIXEL_H = GRID_H * CELL_SIZE # 400
    OFFSET_X = (SCREEN_W - GRID_PIXEL_W) // 2 # 120
    OFFSET_Y = (SCREEN_H - GRID_PIXEL_H) // 2 # 0
    
    MAX_STEPS = 600
    NUM_SOULS = 10
    NUM_SPECTERS = 3
    NUM_TOMBSTONES = 20

    # Colors
    COLOR_BG = (26, 28, 44)
    COLOR_GRID = (42, 45, 61)
    COLOR_TOMBSTONE = (74, 78, 105)
    COLOR_PLAYER = (112, 224, 0)
    COLOR_PLAYER_GLOW = (112, 224, 0, 50)
    COLOR_SOUL = (173, 232, 244)
    COLOR_SOUL_GLOW = (173, 232, 244, 60)
    COLOR_SPECTER = (208, 0, 0)
    COLOR_SPECTER_GLOW = (208, 0, 0, 70)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_PARTICLE = (255, 255, 100)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_W, self.SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        
        self.player_pos = (0, 0)
        self.specters = []
        self.souls = []
        self.tombstones = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.validate_implementation()

    def _generate_path(self, waypoints: list[tuple[int, int]]) -> list[tuple[int, int]]:
        path = []
        for i in range(len(waypoints)):
            start_point = waypoints[i]
            end_point = waypoints[(i + 1) % len(waypoints)]
            
            current_point = list(start_point)
            
            while tuple(current_point) != end_point:
                path.append(tuple(current_point))
                dx = end_point[0] - current_point[0]
                dy = end_point[1] - current_point[1]
                
                if dx != 0:
                    current_point[0] += 1 if dx > 0 else -1
                elif dy != 0:
                    current_point[1] += 1 if dy > 0 else -1
        
        path.append(waypoints[0]) # Add the start point to complete the loop
        return path

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles.clear()
        
        occupied_positions = set()

        self.player_pos = (self.GRID_W // 2, self.GRID_H - 1)
        occupied_positions.add(self.player_pos)

        self.souls.clear()
        while len(self.souls) < self.NUM_SOULS:
            pos = (self.np_random.integers(0, self.GRID_W), self.np_random.integers(0, self.GRID_H - 2))
            if pos not in occupied_positions:
                self.souls.append(pos)
                occupied_positions.add(pos)
        
        self.tombstones.clear()
        while len(self.tombstones) < self.NUM_TOMBSTONES:
            pos = (self.np_random.integers(0, self.GRID_W), self.np_random.integers(0, self.GRID_H))
            if pos not in occupied_positions:
                self.tombstones.append(pos)
                occupied_positions.add(pos)

        self.specters.clear()
        paths = [
            self._generate_path([(2, 2), (13, 2), (13, 5), (2, 5)]),
            self._generate_path([(1, 7), (7, 7), (7, 11), (1, 11)]),
            self._generate_path([(8, 7), (14, 7), (14, 11), (8, 11)]),
        ]
        for i in range(self.NUM_SPECTERS):
            path = paths[i]
            path_start_index = self.np_random.integers(0, len(path))
            self.specters.append({
                "path": path,
                "path_index": path_start_index,
                "pos": path[path_start_index]
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0.1  # Survival reward
        
        dist_before = self._get_dist_to_nearest_soul()
        
        px, py = self.player_pos
        if movement == 1: py -= 1
        elif movement == 2: py += 1
        elif movement == 3: px -= 1
        elif movement == 4: px += 1

        px = max(0, min(self.GRID_W - 1, px))
        py = max(0, min(self.GRID_H - 1, py))
        self.player_pos = (px, py)
        
        dist_after = self._get_dist_to_nearest_soul()

        if dist_after is not None and dist_before is not None:
            if dist_after > dist_before:
                reward -= 2.0

        for specter in self.specters:
            specter["path_index"] = (specter["path_index"] + 1) % len(specter["path"])
            specter["pos"] = specter["path"][specter["path_index"]]

        if self.player_pos in self.souls:
            self.souls.remove(self.player_pos)
            self.score += 10 # Brief says score increases by 10
            reward += 10.0
            # sfx: Soul collect sound
            self._spawn_particles(self.player_pos, 20)
        
        self.score = len(self.souls) # Re-reading brief, victory is collecting 10 souls, so score is just # collected
        self.score = self.NUM_SOULS - len(self.souls)

        self.steps += 1

        terminated = False
        for specter in self.specters:
            if self.player_pos == specter["pos"]:
                self.game_over = True
                terminated = True
                reward = -100.0
                # sfx: Player death sound
                break
        
        if not self.souls:
            self.game_over = True
            terminated = True
            reward = 100.0
            # sfx: Victory sound
        
        if self.steps >= self.MAX_STEPS and not terminated:
            self.game_over = True
            terminated = True
            reward = 100.0
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_dist_to_nearest_soul(self):
        if not self.souls:
            return None
        px, py = self.player_pos
        min_dist = float('inf')
        for sx, sy in self.souls:
            dist = abs(px - sx) + abs(py - sy)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _grid_to_pixel(self, grid_pos):
        gx, gy = grid_pos
        px = self.OFFSET_X + gx * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.OFFSET_Y + gy * self.CELL_SIZE + self.CELL_SIZE // 2
        return int(px), int(py)

    def _spawn_particles(self, grid_pos, count):
        px, py = self._grid_to_pixel(grid_pos)
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifespan = self.np_random.integers(10, 20)
            self.particles.append([[px, py], [vx, vy], lifespan])
            
    def _update_and_draw_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p[0][0] += p[1][0]
            p[0][1] += p[1][1]
            p[2] -= 1
            
            if p[2] <= 0:
                self.particles.pop(i)
            else:
                alpha = int(255 * (p[2] / 20))
                color = (*self.COLOR_PARTICLE, alpha)
                pos = (int(p[0][0]), int(p[0][1]))
                s = pygame.Surface((4,4), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (2,2), 2)
                self.screen.blit(s, (pos[0]-2, pos[1]-2))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)

        for x in range(self.GRID_W + 1):
            px = self.OFFSET_X + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.OFFSET_Y), (px, self.OFFSET_Y + self.GRID_PIXEL_H))
        for y in range(self.GRID_H + 1):
            py = self.OFFSET_Y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.OFFSET_X, py), (self.OFFSET_X + self.GRID_PIXEL_W, py))

        for pos in self.tombstones:
            px, py = self._grid_to_pixel(pos)
            rect = pygame.Rect(px - self.CELL_SIZE//4, py - self.CELL_SIZE//3, self.CELL_SIZE//2, self.CELL_SIZE*2//3)
            pygame.draw.rect(self.screen, self.COLOR_TOMBSTONE, rect, border_radius=3)

        glow_radius = self.CELL_SIZE * 0.4 + 3 * math.sin(self.steps * 0.2)
        for pos in self.souls:
            px, py = self._grid_to_pixel(pos)
            pygame.gfxdraw.filled_circle(self.screen, px, py, int(glow_radius), self.COLOR_SOUL_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, px, py, int(self.CELL_SIZE * 0.25), self.COLOR_SOUL)
            pygame.gfxdraw.aacircle(self.screen, px, py, int(self.CELL_SIZE * 0.25), self.COLOR_SOUL)

        bob_offset = 3 * math.sin(self.steps * 0.3)
        for specter in self.specters:
            px, py = self._grid_to_pixel(specter["pos"])
            py += int(bob_offset)
            radius = int(self.CELL_SIZE * 0.4)
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius + 4, self.COLOR_SPECTER_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, self.COLOR_SPECTER)
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_SPECTER)

        if not self.game_over:
            px, py = self._grid_to_pixel(self.player_pos)
            pulse_radius = self.CELL_SIZE * 0.5 + 2 * math.sin(self.steps * 0.4)
            pygame.gfxdraw.filled_circle(self.screen, px, py, int(pulse_radius), self.COLOR_PLAYER_GLOW)
            player_rect = pygame.Rect(0, 0, self.CELL_SIZE * 0.7, self.CELL_SIZE * 0.7)
            player_rect.center = (px, py)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)

        self._update_and_draw_particles()

        score_text = self.font_large.render(f"SOULS: {self.score} / {self.NUM_SOULS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        time_left = max(0, self.MAX_STEPS - self.steps)
        time_text = self.font_large.render(f"STEPS: {time_left}", True, self.COLOR_UI_TEXT)
        text_rect = time_text.get_rect(topright=(self.SCREEN_W - 20, 20))
        self.screen.blit(time_text, text_rect)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        # Score is number of souls collected
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs_from_init = self._get_observation()
        assert test_obs_from_init.shape == (400, 640, 3)
        assert test_obs_from_init.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_W, GameEnv.SCREEN_H))
    pygame.display.set_caption("Specter Collector")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        move_action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: move_action = 1
                elif event.key == pygame.K_DOWN: move_action = 2
                elif event.key == pygame.K_LEFT: move_action = 3
                elif event.key == pygame.K_RIGHT: move_action = 4
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")
                elif event.key == pygame.K_q: running = False
        
        if move_action != 0:
            action = np.array([move_action, 0, 0])
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"Step: {info['steps']}, Action: {action[0]}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Terminated: {terminated}")
            
            if terminated:
                print(f"--- Episode Finished ---")
                print(f"Final Score: {info['score']}, Final Steps: {info['steps']}")
                pygame.time.wait(2000)
                obs, info = env.reset()
                total_reward = 0
                print("--- New Game Started ---")

        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)

    env.close()