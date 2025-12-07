
# Generated: 2025-08-27T19:32:53.193234
# Source Brief: brief_02186.md
# Brief Index: 2186

        
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

    user_guide = (
        "Controls: Use arrow keys to move on the isometric grid. "
        "Avoid the guards and collect all 10 gems before time runs out!"
    )

    game_description = (
        "A fast-paced isometric stealth game. Navigate a maze-like "
        "level, steal 10 gems, and escape before the 60-second timer "
        "expires, all while avoiding patrolling guards."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_W, self.SCREEN_H = 640, 400
        self.FPS = 30
        self.TIME_LIMIT_SECONDS = 60

        # Grid and Isometric Projection
        self.GRID_W, self.GRID_H = 22, 14
        self.TILE_W, self.TILE_H = 32, 16
        self.TILE_W_HALF, self.TILE_H_HALF = self.TILE_W // 2, self.TILE_H // 2
        self.ORIGIN_X = self.SCREEN_W // 2
        self.ORIGIN_Y = 60

        # Colors
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_PLAYER = (0, 200, 255)
        self.COLOR_PLAYER_GLOW = (0, 100, 150)
        self.COLOR_GUARD = (255, 60, 60)
        self.COLOR_GUARD_GLOW = (150, 30, 30)
        self.COLOR_GUARD_TRAIL = (100, 100, 120)
        self.GEM_COLORS = [(255, 220, 0), (0, 255, 120), (255, 0, 255)]
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TEXT_WARN = (255, 100, 100)
        self.COLOR_OVERLAY = (0, 0, 0, 180)

        # Game Mechanics
        self.NUM_GEMS = 10
        self.NUM_GUARDS = 4
        self.MOVE_FRAMES = 6  # Frames for a single tile move animation
        self.GUARD_TRAIL_LENGTH = 15

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_H, self.SCREEN_W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_W, self.SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_big = pygame.font.SysFont("sans-serif", 48, bold=True)

        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.timer = 0
        self.game_over = False
        self.game_outcome = ""
        self.terminal_reward = 0

        self.player_grid_pos = [0, 0]
        self.player_visual_pos = [0, 0]
        self.player_move_progress = 0.0
        self.player_target_grid_pos = [0, 0]

        self.gems = []
        self.guards = []
        self.particles = []
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.timer = self.FPS * self.TIME_LIMIT_SECONDS
        self.game_over = False
        self.game_outcome = ""
        self.terminal_reward = 0

        self._generate_level()

        self.player_move_progress = 0.0
        self.player_target_grid_pos = list(self.player_grid_pos)
        self.player_visual_pos = self._iso_to_screen(*self.player_grid_pos)

        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1
        self.timer -= 1

        if not self.game_over:
            movement = action[0]
            self._update_player(movement)
            self._update_guards()
            self._update_particles()
            reward = self._check_interactions_and_get_reward()
        
        terminated = self.game_over or self.timer <= 0
        if terminated and not self.game_over: # Handle timeout
            self.game_over = True
            self.game_outcome = "TIME UP"
            self.terminal_reward = -10.0

        final_reward = reward + self.terminal_reward if terminated else reward

        return (
            self._get_observation(),
            final_reward,
            terminated,
            False,
            self._get_info(),
        )

    def _generate_level(self):
        # Place player
        self.player_grid_pos = [self.GRID_W // 2, self.GRID_H - 2]
        
        # Generate gems
        self.gems = []
        occupied_tiles = {tuple(self.player_grid_pos)}
        while len(self.gems) < self.NUM_GEMS:
            gx = self.np_random.integers(1, self.GRID_W - 1)
            gy = self.np_random.integers(1, self.GRID_H - 1)
            if (gx, gy) not in occupied_tiles:
                self.gems.append({
                    "pos": [gx, gy],
                    "color": random.choice(self.GEM_COLORS),
                    "bob_offset": self.np_random.uniform(0, math.pi * 2)
                })
                occupied_tiles.add((gx, gy))

        # Generate guards
        self.guards = []
        for _ in range(self.NUM_GUARDS):
            path_type = self.np_random.integers(0, 2)
            if path_type == 0: # Horizontal patrol
                y = self.np_random.integers(1, self.GRID_H - 2)
                x1 = self.np_random.integers(1, self.GRID_W // 2 - 1)
                x2 = self.np_random.integers(self.GRID_W // 2 + 1, self.GRID_W - 2)
                path = [[x1, y], [x2, y]]
            else: # Vertical patrol
                x = self.np_random.integers(1, self.GRID_W - 2)
                y1 = self.np_random.integers(1, self.GRID_H // 2 - 1)
                y2 = self.np_random.integers(self.GRID_H // 2 + 1, self.GRID_H - 2)
                path = [[x, y1], [x, y2]]
            
            start_pos = list(path[0])
            self.guards.append({
                "grid_pos": start_pos,
                "visual_pos": self._iso_to_screen(*start_pos),
                "target_grid_pos": list(path[0]),
                "move_progress": 0.0,
                "path": path,
                "path_index": 1,
                "trail": [],
            })

    def _update_player(self, movement):
        if self.player_move_progress > 0:
            self.player_move_progress += 1.0 / self.MOVE_FRAMES
            if self.player_move_progress >= 1.0:
                self.player_move_progress = 0.0
                self.player_grid_pos = list(self.player_target_grid_pos)
        
        if self.player_move_progress == 0.0 and movement != 0:
            target = list(self.player_grid_pos)
            if movement == 1: target[1] -= 1  # Up -> Up-Left
            elif movement == 2: target[1] += 1 # Down -> Down-Right
            elif movement == 3: target[0] -= 1 # Left -> Down-Left
            elif movement == 4: target[0] += 1 # Right -> Up-Right

            if 0 <= target[0] < self.GRID_W and 0 <= target[1] < self.GRID_H:
                self.player_target_grid_pos = target
                self.player_move_progress = 1.0 / self.MOVE_FRAMES
        
        start_pos = self._iso_to_screen(*self.player_grid_pos)
        end_pos = self._iso_to_screen(*self.player_target_grid_pos)
        progress = min(1.0, self.player_move_progress)
        self.player_visual_pos[0] = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
        self.player_visual_pos[1] = start_pos[1] + (end_pos[1] - start_pos[1]) * progress

    def _update_guards(self):
        for guard in self.guards:
            if guard["move_progress"] > 0:
                guard["move_progress"] += 1.0 / (self.MOVE_FRAMES * 1.5) # Guards move slower
                if guard["move_progress"] >= 1.0:
                    guard["move_progress"] = 0.0
                    guard["grid_pos"] = list(guard["target_grid_pos"])
            
            if guard["move_progress"] == 0.0:
                current_target_node = guard["path"][guard["path_index"]]
                if guard["grid_pos"] == current_target_node:
                    guard["path_index"] = (guard["path_index"] + 1) % len(guard["path"])
                
                next_node = guard["path"][guard["path_index"]]
                dx = np.sign(next_node[0] - guard["grid_pos"][0])
                dy = np.sign(next_node[1] - guard["grid_pos"][1])

                guard["target_grid_pos"][0] = guard["grid_pos"][0] + dx
                guard["target_grid_pos"][1] = guard["grid_pos"][1] + dy
                guard["move_progress"] = 1.0 / (self.MOVE_FRAMES * 1.5)

            start_pos = self._iso_to_screen(*guard["grid_pos"])
            end_pos = self._iso_to_screen(*guard["target_grid_pos"])
            progress = min(1.0, guard["move_progress"])
            guard["visual_pos"][0] = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
            guard["visual_pos"][1] = start_pos[1] + (end_pos[1] - start_pos[1]) * progress

            guard["trail"].append(list(guard["visual_pos"]))
            if len(guard["trail"]) > self.GUARD_TRAIL_LENGTH:
                guard["trail"].pop(0)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # Gravity
            p["life"] -= 1

    def _check_interactions_and_get_reward(self):
        reward = 0.01  # Small reward for surviving a step

        # Gem collection
        for gem in self.gems[:]:
            if self.player_grid_pos == gem["pos"]:
                self.gems.remove(gem)
                self.score += 1
                reward += 10.0
                # Sound: gem_collect.wav
                # Spawn particles
                gem_screen_pos = self._iso_to_screen(*gem["pos"])
                for _ in range(15):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    speed = self.np_random.uniform(1, 4)
                    self.particles.append({
                        "pos": list(gem_screen_pos),
                        "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                        "life": self.np_random.integers(15, 30),
                        "color": gem["color"],
                        "size": self.np_random.integers(2, 5)
                    })
                break

        # Guard detection
        for guard in self.guards:
            dist_x = abs(self.player_grid_pos[0] - guard["grid_pos"][0])
            dist_y = abs(self.player_grid_pos[1] - guard["grid_pos"][1])
            if dist_x < 1 and dist_y < 1: # On same tile
                self.game_over = True
                self.game_outcome = "CAUGHT!"
                self.terminal_reward = -100.0
                # Sound: player_detected.wav
                break
        
        # Win condition
        if self.score >= self.NUM_GEMS:
            self.game_over = True
            self.game_outcome = "YOU WIN!"
            self.terminal_reward = 100.0
            # Sound: victory.wav

        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                px, py = self._iso_to_screen(x, y)
                points = [
                    (px, py),
                    (px + self.TILE_W_HALF, py + self.TILE_H_HALF),
                    (px, py + self.TILE_H),
                    (px - self.TILE_W_HALF, py + self.TILE_H_HALF)
                ]
                pygame.draw.polygon(self.screen, self.COLOR_GRID, points, 1)

    def _render_game_elements(self):
        # Render order: trails, gems, guards, player, particles
        
        # Guard trails
        for guard in self.guards:
            for i, pos in enumerate(guard["trail"]):
                alpha = int(100 * (i / self.GUARD_TRAIL_LENGTH))
                s = pygame.Surface((self.TILE_W, self.TILE_H), pygame.SRCALPHA)
                pygame.draw.circle(s, self.COLOR_GUARD_TRAIL + (alpha,), (self.TILE_W_HALF, self.TILE_H_HALF), 3)
                self.screen.blit(s, (int(pos[0] - self.TILE_W_HALF), int(pos[1] - self.TILE_H_HALF)))

        # Gems
        for gem in self.gems:
            px, py = self._iso_to_screen(*gem["pos"])
            bob = math.sin(gem["bob_offset"] + self.steps * 0.1) * 3
            py += bob
            
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), 10, (*gem["color"], 50))
            pygame.gfxdraw.aacircle(self.screen, int(px), int(py), 10, (*gem["color"], 100))
            
            # Gem shape
            points = [(px, py - 6), (px + 5, py), (px, py + 6), (px - 5, py)]
            pygame.gfxdraw.aapolygon(self.screen, points, gem["color"])
            pygame.gfxdraw.filled_polygon(self.screen, points, gem["color"])

        # Guards
        for guard in self.guards:
            px, py = guard["visual_pos"]
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, int(px), int(py - 4), 12, (*self.COLOR_GUARD_GLOW, 100))
            pygame.gfxdraw.aacircle(self.screen, int(px), int(py - 4), 12, (*self.COLOR_GUARD_GLOW, 150))
            # Body
            pygame.draw.circle(self.screen, self.COLOR_GUARD, (int(px), int(py - 4)), 7)

        # Player
        px, py = self.player_visual_pos
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, int(px), int(py - 4), 14, (*self.COLOR_PLAYER_GLOW, 100))
        pygame.gfxdraw.aacircle(self.screen, int(px), int(py - 4), 14, (*self.COLOR_PLAYER_GLOW, 150))
        # Body
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, (int(px), int(py - 4)), 8)
        
        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / 30.0))))
            pygame.draw.circle(self.screen, p["color"] + (alpha,), p["pos"], int(p["size"]))

    def _render_ui(self):
        # Gem Score
        gem_text = self.font_ui.render(f"GEMS: {self.score}/{self.NUM_GEMS}", True, self.COLOR_TEXT)
        self.screen.blit(gem_text, (10, 10))

        # Timer
        time_left = self.timer / self.FPS
        time_color = self.COLOR_TEXT_WARN if time_left < 10 else self.COLOR_TEXT
        time_text = self.font_ui.render(f"TIME: {time_left:.2f}", True, time_color)
        time_rect = time_text.get_rect(topright=(self.SCREEN_W - 10, 10))
        self.screen.blit(time_text, time_rect)

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_W, self.SCREEN_H), pygame.SRCALPHA)
        overlay.fill(self.COLOR_OVERLAY)
        self.screen.blit(overlay, (0, 0))

        text_surface = self.font_big.render(self.game_outcome, True, self.COLOR_TEXT)
        text_rect = text_surface.get_rect(center=(self.SCREEN_W / 2, self.SCREEN_H / 2))
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "time_left": round(self.timer / self.FPS, 2)}

    def _iso_to_screen(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_W_HALF
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_H_HALF
        return [screen_x, screen_y]

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_H, self.SCREEN_W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen_w, screen_h = obs.shape[1], obs.shape[0]
    display_screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Isometric Gem Stealer")
    
    running = True
    terminated = False
    
    while running:
        # Player controls
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                terminated = False

        if terminated:
            # On game over, wait for reset
            pass
        else:
            obs, reward, terminated, truncated, info = env.step(action)
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    env.close()