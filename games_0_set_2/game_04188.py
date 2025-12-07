
# Generated: 2025-08-28T01:40:19.255756
# Source Brief: brief_04188.md
# Brief Index: 4188

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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

    user_guide = (
        "Controls: Arrow keys to move. Hold Space while moving into a crystal to push it."
    )

    game_description = (
        "Navigate a procedurally generated isometric cavern. Push crystals to clear paths, avoid traps, and reach the green exit tile."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Configuration ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 20, 15
        self.MAX_STEPS = 1000

        # --- Visuals ---
        self.T_W, self.T_H = 40, 20  # Isometric tile dimensions
        self.CUBE_H = 20 # Visual height of player/crystal cubes
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = self.HEIGHT // 2 - self.GRID_HEIGHT * self.T_H // 2 + 20

        # --- Colors ---
        self.COLOR_BG = (26, 28, 44)
        self.COLOR_FLOOR = (50, 52, 70)
        self.COLOR_FLOOR_ACCENT = (60, 62, 80)
        self.COLOR_WALL = (35, 38, 55)
        self.COLOR_WALL_TOP = (80, 85, 110)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_ACCENT = (150, 220, 255)
        self.COLOR_CRYSTAL = (150, 80, 255)
        self.COLOR_CRYSTAL_ACCENT = (200, 160, 255)
        self.COLOR_TRAP = (255, 50, 50)
        self.COLOR_EXIT = (50, 255, 150)
        self.COLOR_SHADOW = (0, 0, 0, 80)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_TEXT_SHADOW = (20, 20, 20)

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
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        # --- Game State ---
        self.player_pos = (0, 0)
        self.crystals = []
        self.traps = []
        self.exit_pos = (0, 0)
        self.walls = set()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        
        # --- Difficulty Progression ---
        self.successful_escapes = 0
        self.initial_crystals = 10
        self.initial_traps = 3
        
        self.reset()
        self.validate_implementation()

    def _get_difficulty_params(self):
        num_crystals = self.initial_crystals + (self.successful_escapes // 2)
        num_traps = self.initial_traps + (self.successful_escapes // 3)
        return num_crystals, num_traps

    def _generate_level(self):
        self.walls.clear()
        self.crystals.clear()
        self.traps.clear()
        
        # 1. Create wall boundaries
        for x in range(-1, self.GRID_WIDTH + 1):
            self.walls.add((x, -1))
            self.walls.add((x, self.GRID_HEIGHT))
        for y in range(0, self.GRID_HEIGHT):
            self.walls.add((-1, y))
            self.walls.add((self.GRID_WIDTH, y))

        # 2. Place player and exit
        all_tiles = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.player_pos, self.exit_pos = self.np_random.choice(len(all_tiles), 2, replace=False)
        self.player_pos = tuple(all_tiles[self.player_pos])
        self.exit_pos = tuple(all_tiles[self.exit_pos])
        
        # 3. Find a guaranteed path to ensure solvability
        path = self._bfs_path(self.player_pos, self.exit_pos, self.walls)
        if path is None: # Should not happen on an empty grid, but as a safeguard
            self.reset()
            return
        
        # 4. Place crystals and traps
        num_crystals, num_traps = self._get_difficulty_params()
        
        occupied = set(path) | self.walls | {self.player_pos, self.exit_pos}
        available_tiles = [t for t in all_tiles if t not in occupied]
        
        if len(available_tiles) < num_crystals + num_traps: # Not enough space, regenerate
            self.reset()
            return
            
        placements = self.np_random.choice(len(available_tiles), num_crystals + num_traps, replace=False)
        
        for i in range(num_crystals):
            self.crystals.append(available_tiles[placements[i]])
        
        for i in range(num_crystals, num_crystals + num_traps):
            self.traps.append(available_tiles[placements[i]])

    def _bfs_path(self, start, end, obstacles):
        q = deque([(start, [start])])
        visited = {start}
        while q:
            (node, path) = q.popleft()
            if node == end:
                return path
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (node[0] + dx, node[1] + dy)
                if neighbor not in visited and neighbor not in obstacles:
                    visited.add(neighbor)
                    q.append((neighbor, path + [neighbor]))
        return None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles.clear()
        self._generate_level()
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        is_pushing = action[1] == 1
        reward = -0.1  # Per-step penalty
        self.steps += 1
        self.particles.clear()

        # --- Action Logic ---
        if movement > 0:
            dx = (0, 0, 0, -1, 1)[movement]
            dy = (0, -1, 1, 0, 0)[movement]
            px, py = self.player_pos
            target_pos = (px + dx, py + dy)

            if target_pos in self.walls:
                pass # Action failed (hit wall)
            elif target_pos in self.crystals:
                if is_pushing:
                    push_target_pos = (target_pos[0] + dx, target_pos[1] + dy)
                    if push_target_pos not in self.walls and push_target_pos not in self.crystals:
                        # Successful push
                        self.crystals.remove(target_pos)
                        self.crystals.append(push_target_pos)
                        self.player_pos = target_pos
                        # SFX: Crystal slide
                        self._spawn_particles(target_pos, self.COLOR_CRYSTAL_ACCENT, 10)
                else:
                    pass # Action failed (bumped into crystal without pushing)
            elif target_pos == self.exit_pos:
                self.player_pos = target_pos
                reward = 100.0
                self.game_over = True
                self.successful_escapes += 1
                # SFX: Win jingle
                self._spawn_particles(target_pos, self.COLOR_EXIT, 30, 20)
            elif target_pos in self.traps:
                self.player_pos = target_pos
                reward = -10.0
                self.game_over = True
                # SFX: Trap spring/explosion
                self._spawn_particles(target_pos, self.COLOR_TRAP, 50, 15)
            else: # Empty space
                self.player_pos = target_pos

        self.score += reward
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.steps >= self.MAX_STEPS and not self.game_over:
            terminated = True
            
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _iso_to_screen(self, x, y):
        sx = self.ORIGIN_X + (x - y) * self.T_W / 2
        sy = self.ORIGIN_Y + (x + y) * self.T_H / 2
        return int(sx), int(sy)

    def _draw_iso_cube(self, surface, color, light_color, x, y, z_offset=0):
        sx, sy = self._iso_to_screen(x, y)
        sy -= z_offset
        
        # Points for the cube
        p_top_mid = (sx, sy)
        p_top_l = (sx - self.T_W / 2, sy + self.T_H / 2)
        p_top_r = (sx + self.T_W / 2, sy + self.T_H / 2)
        p_top_bot = (sx, sy + self.T_H)
        
        p_bot_mid = (sx, sy + self.CUBE_H)
        p_bot_l = (sx - self.T_W / 2, sy + self.T_H / 2 + self.CUBE_H)
        p_bot_r = (sx + self.T_W / 2, sy + self.T_H / 2 + self.CUBE_H)

        # Draw faces
        darker_color = tuple(max(0, c - 40) for c in color)
        pygame.gfxdraw.filled_polygon(surface, [p_top_l, p_top_mid, p_bot_mid, p_bot_l], darker_color) # Left face
        pygame.gfxdraw.filled_polygon(surface, [p_top_r, p_top_mid, p_bot_mid, p_bot_r], darker_color) # Right face
        pygame.gfxdraw.filled_polygon(surface, [p_top_l, p_top_mid, p_top_r, p_top_bot], light_color) # Top face

        # Draw outlines for clarity
        outline_color = tuple(max(0, c - 60) for c in color)
        pygame.draw.lines(surface, outline_color, False, [p_bot_l, p_top_l, p_top_mid, p_top_r, p_bot_r], 1)
        pygame.draw.line(surface, outline_color, p_top_mid, p_top_bot, 1)

    def _spawn_particles(self, pos, color, count, speed_mult=10):
        sx, sy = self._iso_to_screen(pos[0], pos[1])
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * (speed_mult / 10.0)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append([sx, sy, vel[0], vel[1], self.np_random.integers(1,4), color])

    def _render_game(self):
        # --- Create a sorted list of all drawable entities ---
        # Sort key is render depth (y then x)
        draw_list = []
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                draw_list.append(('floor', (x, y)))
        for trap_pos in self.traps:
            draw_list.append(('trap', trap_pos))
        for crystal_pos in self.crystals:
            draw_list.append(('crystal', crystal_pos))
        
        draw_list.append(('exit', self.exit_pos))
        draw_list.append(('player', self.player_pos))
        
        # Sort by grid y, then grid x for correct isometric rendering
        draw_list.sort(key=lambda item: (item[1][0] + item[1][1], item[1][1]))

        # --- Draw all entities ---
        for item_type, pos in draw_list:
            gx, gy = pos
            sx, sy = self._iso_to_screen(gx, gy)
            
            # Shadow surface for transparency
            shadow_surf = pygame.Surface((self.T_W, self.T_H + self.CUBE_H), pygame.SRCALPHA)
            
            if item_type == 'floor':
                color = self.COLOR_FLOOR_ACCENT if (gx + gy) % 2 == 0 else self.COLOR_FLOOR
                pygame.gfxdraw.filled_polygon(self.screen, [
                    (sx, sy + self.T_H/2), (sx + self.T_W/2, sy + self.T_H), 
                    (sx, sy + self.T_H * 1.5), (sx - self.T_W/2, sy + self.T_H)
                ], color)
            elif item_type == 'trap':
                pygame.gfxdraw.filled_circle(self.screen, sx, int(sy + self.T_H), int(self.T_W/4), self.COLOR_TRAP)
                pygame.gfxdraw.aacircle(self.screen, sx, int(sy + self.T_H), int(self.T_W/4), self.COLOR_TRAP)
            elif item_type == 'exit':
                glow_color = list(self.COLOR_EXIT) + [100]
                pygame.gfxdraw.filled_circle(self.screen, sx, int(sy + self.T_H), int(self.T_W/3), glow_color)
                pygame.gfxdraw.filled_circle(self.screen, sx, int(sy + self.T_H), int(self.T_W/4), self.COLOR_EXIT)
            elif item_type == 'crystal':
                pygame.draw.ellipse(shadow_surf, self.COLOR_SHADOW, [0, self.CUBE_H, self.T_W, self.T_H])
                self.screen.blit(shadow_surf, (sx - self.T_W/2, sy + self.T_H/2 - self.CUBE_H))
                self._draw_iso_cube(self.screen, self.COLOR_CRYSTAL, self.COLOR_CRYSTAL_ACCENT, gx, gy)
            elif item_type == 'player':
                pygame.draw.ellipse(shadow_surf, self.COLOR_SHADOW, [0, self.CUBE_H, self.T_W, self.T_H])
                self.screen.blit(shadow_surf, (sx - self.T_W/2, sy + self.T_H/2 - self.CUBE_H))
                self._draw_iso_cube(self.screen, self.COLOR_PLAYER, self.COLOR_PLAYER_ACCENT, gx, gy)
                # Player glow
                pygame.gfxdraw.filled_circle(self.screen, sx, int(sy + self.T_H/2), int(self.T_W/2.5), (200, 220, 255, 30))

        # --- Draw Walls ---
        for wx, wy in self.walls:
            if 0 <= wx < self.GRID_WIDTH and 0 <= wy < self.GRID_HEIGHT:
                 self._draw_iso_cube(self.screen, self.COLOR_WALL, self.COLOR_WALL_TOP, wx, wy)

        # --- Draw Particles ---
        for p in self.particles[:]:
            p[0] += p[2]
            p[1] += p[3]
            p[4] -= 0.1 # lifetime
            if p[4] <= 0:
                self.particles.remove(p)
            else:
                pygame.draw.circle(self.screen, p[5], (int(p[0]), int(p[1])), int(p[4]))


    def _render_ui(self):
        def draw_text(text, pos, font, color, shadow_color):
            text_surf = font.render(text, True, color)
            shadow_surf = font.render(text, True, shadow_color)
            self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(text_surf, pos)

        score_text = f"SCORE: {int(self.score)}"
        steps_text = f"STEPS: {self.steps}/{self.MAX_STEPS}"
        
        draw_text(steps_text, (10, 10), self.font_large, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        draw_text(score_text, (self.WIDTH - self.font_large.size(score_text)[0] - 10, 10), self.font_large, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        if self.game_over:
            msg = "LEVEL COMPLETE" if self.score > 0 else "GAME OVER"
            msg_size = self.font_large.size(msg)
            draw_text(msg, (self.WIDTH/2 - msg_size[0]/2, self.HEIGHT/2 - msg_size[1]/2), self.font_large, self.COLOR_EXIT if self.score > 0 else self.COLOR_TRAP, self.COLOR_TEXT_SHADOW)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "escapes": self.successful_escapes}

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # --- Manual Play ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    
    print(env.user_guide)

    while not terminated:
        # --- Pygame Event Handling for Manual Control ---
        action = [0, 0, 0] # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    print("--- LEVEL RESET ---")
                    continue

                keys = pygame.key.get_pressed()
                if keys[pygame.K_SPACE]:
                    action[1] = 1
                if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                    action[2] = 1
                
                # --- Step the environment only on key press ---
                if action[0] != 0:
                    obs, reward, term, trunc, info = env.step(action)
                    terminated = term
                    print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Steps: {info['steps']}")

        # --- Render to screen ---
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"--- GAME OVER --- Final Score: {info['score']:.2f}")
            pygame.time.wait(2000) # Pause before resetting
            obs, info = env.reset()
            terminated = False

    env.close()