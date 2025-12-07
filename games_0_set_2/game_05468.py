
# Generated: 2025-08-28T05:07:16.473244
# Source Brief: brief_05468.md
# Brief Index: 5468

        
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
        "Controls: Use arrow keys to move the cursor. Press Shift to cycle crystal types. "
        "Press Space to place the selected crystal at the cursor's location."
    )

    game_description = (
        "A strategic puzzle game. Navigate a crystal cavern and place crystals to redirect "
        "light beams, illuminating all targets to win before you run out of resources."
    )

    auto_advance = False

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 32, 24
    MAX_STEPS = 500
    STARTING_CRYSTALS = [10, 5]  # [Refractors, Splitters]

    # --- Colors ---
    COLOR_BG = (15, 10, 35)
    COLOR_GRID = (30, 25, 55)
    COLOR_WALL = (50, 45, 80)
    COLOR_WALL_TOP = (80, 75, 110)
    COLOR_LIGHT_BEAM = (255, 255, 200)
    COLOR_LIGHT_GLOW = (255, 255, 150, 50)
    COLOR_TARGET = (70, 70, 90)
    COLOR_TARGET_LIT = (255, 220, 0)
    COLOR_TARGET_LIT_GLOW = (255, 220, 0, 100)
    COLOR_CURSOR = (255, 255, 255)
    CRYSTAL_COLORS = [
        {"main": (255, 80, 0), "glow": (255, 80, 0, 150)},    # Refractor
        {"main": (0, 180, 255), "glow": (0, 180, 255, 150)},  # Splitter
    ]
    TEXT_COLOR = (220, 220, 240)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 20)
        self.font_large = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 48)

        self.iso_tile_w = self.WIDTH / (self.GRID_WIDTH + 2)
        self.iso_tile_h = self.iso_tile_w * 0.5
        self.origin_x = self.WIDTH / 2
        self.origin_y = self.HEIGHT * 0.2

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.rng = None
        self.cursor_pos = None
        self.walls = None
        self.light_source = None
        self.targets = None
        self.placed_crystals = None
        self.crystal_inventory = None
        self.selected_crystal_type = None
        self.light_paths = None
        self.particles = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        
        self.cursor_pos = np.array([self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2])
        self.placed_crystals = {}
        self.crystal_inventory = list(self.STARTING_CRYSTALS)
        self.selected_crystal_type = 0
        self.particles = []

        self._procedural_generation()
        self._calculate_light_paths()
        self._update_targets()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- Action Handling ---
        # 1. Move cursor
        if movement > 0:
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][movement - 1]
            new_pos = self.cursor_pos + np.array([dx, dy])
            if 0 <= new_pos[0] < self.GRID_WIDTH and 0 <= new_pos[1] < self.GRID_HEIGHT:
                self.cursor_pos = new_pos

        # 2. Cycle crystal type
        if shift_held:
            self.selected_crystal_type = (self.selected_crystal_type + 1) % len(self.crystal_inventory)
            # Sound: crystal_select.wav

        # 3. Place crystal
        placed_crystal_this_step = False
        cursor_tuple = tuple(self.cursor_pos)
        if space_held and cursor_tuple not in self.walls and cursor_tuple not in self.placed_crystals:
            if self.crystal_inventory[self.selected_crystal_type] > 0:
                self.crystal_inventory[self.selected_crystal_type] -= 1
                self.placed_crystals[cursor_tuple] = self.selected_crystal_type
                placed_crystal_this_step = True
                reward -= 0.1
                self._add_particles(cursor_tuple, self.CRYSTAL_COLORS[self.selected_crystal_type]["main"])
                # Sound: crystal_place.wav
        
        # --- Game Logic Update ---
        if placed_crystal_this_step:
            self._calculate_light_paths()
            newly_lit = self._update_targets()
            reward += newly_lit * 1.0
            # Sound: target_lit.wav if newly_lit > 0

        self.steps += 1
        self._update_particles()
        
        # --- Termination Check ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            all_lit = all(t['lit'] for t in self.targets.values())
            if all_lit:
                self.win_message = "ALL TARGETS ILLUMINATED"
                reward += 10.0
                # Sound: win.wav
                if sum(self.crystal_inventory) >= sum(self.STARTING_CRYSTALS) / 2:
                    reward += 50.0
                    self.win_message = "EFFICIENT VICTORY!"
            elif sum(self.crystal_inventory) == 0:
                self.win_message = "OUT OF CRYSTALS"
                reward -= 10.0
                # Sound: lose.wav
            else: # Max steps
                self.win_message = "TIME UP"

        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _procedural_generation(self):
        self.walls = set()
        for i in range(self.GRID_WIDTH):
            self.walls.add((i, -1))
            self.walls.add((i, self.GRID_HEIGHT))
        for i in range(self.GRID_HEIGHT):
            self.walls.add((-1, i))
            self.walls.add((self.GRID_WIDTH, i))

        num_internal_walls = self.rng.integers(5, 15)
        for _ in range(num_internal_walls):
            start_x = self.rng.integers(0, self.GRID_WIDTH)
            start_y = self.rng.integers(0, self.GRID_HEIGHT)
            length = self.rng.integers(2, 6)
            direction = self.rng.choice(['h', 'v'])
            for i in range(length):
                x = start_x + i if direction == 'h' else start_x
                y = start_y if direction == 'h' else start_y + i
                if 0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT:
                    self.walls.add((x, y))

        source_pos = (self.rng.integers(self.GRID_WIDTH // 4, self.GRID_WIDTH * 3 // 4), -1)
        self.light_source = {'pos': source_pos, 'dir': (0, 1)}

        self.targets = {}
        num_targets = self.rng.integers(3, 6)
        while len(self.targets) < num_targets:
            pos = (self.rng.integers(0, self.GRID_WIDTH), self.rng.integers(0, self.GRID_HEIGHT))
            if pos not in self.walls and pos not in self.targets and pos != self.light_source['pos']:
                self.targets[pos] = {'lit': False, 'anim_prog': 0.0}

    def _calculate_light_paths(self):
        self.light_paths = []
        q = deque([(self.light_source['pos'], self.light_source['dir'])])
        visited_crystal_dirs = set()
        max_beams = 50

        while q and len(self.light_paths) < max_beams:
            start_pos, direction = q.popleft()
            
            key = (start_pos, direction)
            if key in visited_crystal_dirs: continue
            if start_pos in self.placed_crystals: visited_crystal_dirs.add(key)

            path = [start_pos]
            current_pos = np.array(start_pos, dtype=float)
            
            for _ in range(max(self.GRID_WIDTH, self.GRID_HEIGHT) * 2):
                current_pos += np.array(direction, dtype=float) * 0.5
                grid_pos = (int(round(current_pos[0])), int(round(current_pos[1])))

                if grid_pos in self.walls:
                    path.append(grid_pos)
                    break
                
                if grid_pos in self.placed_crystals:
                    path.append(grid_pos)
                    crystal_type = self.placed_crystals[grid_pos]
                    
                    # Type 0: Refractor (90 degree bounce)
                    if crystal_type == 0:
                        new_dir = (-direction[1], direction[0]) if (direction[0] + direction[1]) % 2 != 0 else (direction[1], -direction[0])
                        q.append((grid_pos, new_dir))
                    # Type 1: Splitter
                    elif crystal_type == 1:
                        q.append((grid_pos, (direction[1], direction[0])))
                        q.append((grid_pos, (-direction[1], -direction[0])))
                    break
                
                if not (0 <= grid_pos[0] < self.GRID_WIDTH and 0 <= grid_pos[1] < self.GRID_HEIGHT):
                    path.append(grid_pos)
                    break
            else: # No collision, path goes to edge
                final_pos = np.array(start_pos) + np.array(direction) * max(self.GRID_WIDTH, self.GRID_HEIGHT) * 2
                path.append(tuple(final_pos))

            self.light_paths.append(path)

    def _update_targets(self):
        newly_lit = 0
        
        # Reset all targets to unlit before checking
        for pos in self.targets:
            self.targets[pos]['lit'] = False

        for path in self.light_paths:
            for i in range(len(path) - 1):
                p1 = np.array(path[i])
                p2 = np.array(path[i+1])
                
                for t_pos, t_data in self.targets.items():
                    p_target = np.array(t_pos)
                    # Check if target is on the line segment
                    dist = np.linalg.norm(np.cross(p2-p1, p1-p_target)) / (np.linalg.norm(p2-p1) + 1e-8)
                    if dist < 0.5 and np.dot(p_target-p1, p2-p1) >= 0 and np.dot(p_target-p2, p1-p2) >= 0:
                        if not t_data['lit']:
                            newly_lit += 1
                        t_data['lit'] = True
        return newly_lit

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            return True
        if all(t['lit'] for t in self.targets.values()):
            return True
        if sum(self.crystal_inventory) == 0 and not all(t['lit'] for t in self.targets.values()):
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_HEIGHT + 1):
            p1 = self._iso_to_screen(0, y)
            p2 = self._iso_to_screen(self.GRID_WIDTH, y)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)
        for x in range(self.GRID_WIDTH + 1):
            p1 = self._iso_to_screen(x, 0)
            p2 = self._iso_to_screen(x, self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)

        # Draw elements in isometric order
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                pos = (x, y)
                if pos in self.walls:
                    self._draw_iso_cube(self.screen, pos, self.COLOR_WALL, self.COLOR_WALL_TOP)
                if pos in self.targets:
                    self._draw_target(pos, self.targets[pos])
        
        # Draw crystals
        for pos, type in self.placed_crystals.items():
            self._draw_crystal(pos, type)

        # Draw light source
        self._draw_iso_cube(self.screen, self.light_source['pos'], self.COLOR_LIGHT_GLOW, self.COLOR_LIGHT_BEAM)

        # Draw light beams
        self._draw_light_beams()

        # Draw cursor
        self._draw_cursor()
        
        # Draw particles
        for p in self.particles:
            p_screen = self._iso_to_screen(p['pos'][0], p['pos'][1])
            pygame.draw.circle(self.screen, p['color'], p_screen, int(p['size']))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score:.1f}", True, self.TEXT_COLOR)
        self.screen.blit(score_text, (10, 10))

        # Crystal inventory
        ui_x = 15
        ui_y = 40
        for i, count in enumerate(self.crystal_inventory):
            color = self.CRYSTAL_COLORS[i]["main"]
            pygame.draw.rect(self.screen, color, (ui_x, ui_y, 20, 20), border_radius=3)
            if self.selected_crystal_type == i:
                pygame.draw.rect(self.screen, self.COLOR_CURSOR, (ui_x-2, ui_y-2, 24, 24), 2, border_radius=4)
            
            count_text = self.font_large.render(f"x {count}", True, self.TEXT_COLOR)
            self.screen.blit(count_text, (ui_x + 28, ui_y))
            ui_y += 30

        # Steps
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.TEXT_COLOR)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))
        
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_title.render(self.win_message, True, self.COLOR_TARGET_LIT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _iso_to_screen(self, x, y):
        screen_x = self.origin_x + (x - y) * self.iso_tile_w / 2
        screen_y = self.origin_y + (x + y) * self.iso_tile_h / 2
        return int(screen_x), int(screen_y)

    def _draw_iso_cube(self, surface, pos, side_color, top_color):
        x, y = pos
        px, py = self._iso_to_screen(x, y)
        tile_w, tile_h = self.iso_tile_w, self.iso_tile_h
        
        points_top = [
            (px, py - tile_h),
            (px + tile_w / 2, py - tile_h / 2),
            (px, py),
            (px - tile_w / 2, py - tile_h / 2)
        ]
        points_left = [
            (px - tile_w/2, py - tile_h/2), (px, py), (px, py + tile_h), (px - tile_w/2, py + tile_h/2)
        ]
        points_right = [
            (px + tile_w/2, py - tile_h/2), (px, py), (px, py + tile_h), (px + tile_w/2, py + tile_h/2)
        ]
        
        pygame.gfxdraw.filled_polygon(surface, points_left, side_color)
        pygame.gfxdraw.filled_polygon(surface, points_right, side_color)
        pygame.gfxdraw.filled_polygon(surface, points_top, top_color)
        pygame.gfxdraw.aapolygon(surface, points_top, top_color)

    def _draw_target(self, pos, data):
        px, py = self._iso_to_screen(pos[0], pos[1])
        if data['lit']:
            if data['anim_prog'] < 1.0: data['anim_prog'] += 0.1
            glow_rad = int(self.iso_tile_w * 0.5 * data['anim_prog'])
            pygame.gfxdraw.filled_circle(self.screen, px, py, glow_rad, self.COLOR_TARGET_LIT_GLOW)
            pygame.gfxdraw.aacircle(self.screen, px, py, glow_rad, self.COLOR_TARGET_LIT_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, px, py, int(self.iso_tile_w * 0.3), self.COLOR_TARGET_LIT)
        else:
            if data['anim_prog'] > 0.0: data['anim_prog'] -= 0.1
            pygame.gfxdraw.filled_circle(self.screen, px, py, int(self.iso_tile_w * 0.3), self.COLOR_TARGET)
            pygame.gfxdraw.aacircle(self.screen, px, py, int(self.iso_tile_w * 0.3), self.COLOR_TARGET)
            
    def _draw_crystal(self, pos, type):
        px, py = self._iso_to_screen(pos[0], pos[1])
        color_data = self.CRYSTAL_COLORS[type]
        tile_w, tile_h = self.iso_tile_w, self.iso_tile_h
        points = [
            (px, py - tile_h * 0.5),
            (px + tile_w * 0.4, py),
            (px, py + tile_h * 0.5),
            (px - tile_w * 0.4, py)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, color_data["glow"])
        pygame.gfxdraw.filled_polygon(self.screen, points, color_data["main"])
        pygame.gfxdraw.aapolygon(self.screen, points, color_data["main"])

    def _draw_light_beams(self):
        anim_offset = (pygame.time.get_ticks() % 2000) / 2000.0
        for path in self.light_paths:
            for i in range(len(path) - 1):
                p1 = self._iso_to_screen(path[i][0], path[i][1])
                p2 = self._iso_to_screen(path[i+1][0], path[i+1][1])
                pygame.draw.line(self.screen, self.COLOR_LIGHT_GLOW, p1, p2, 7)
                pygame.draw.aaline(self.screen, self.COLOR_LIGHT_BEAM, p1, p2, 1)

                # Particle animation on beam
                dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
                if dist > 0:
                    num_particles = int(dist / 30)
                    for j in range(num_particles):
                        prog = (j / num_particles + anim_offset) % 1.0
                        px = int(p1[0] + (p2[0]-p1[0]) * prog)
                        py = int(p1[1] + (p2[1]-p1[1]) * prog)
                        pygame.draw.circle(self.screen, self.COLOR_LIGHT_BEAM, (px, py), 2)


    def _draw_cursor(self):
        px, py = self._iso_to_screen(self.cursor_pos[0], self.cursor_pos[1])
        tile_w, tile_h = self.iso_tile_w, self.iso_tile_h
        points = [
            (px, py),
            (px + tile_w / 2, py + tile_h / 2),
            (px, py + tile_h),
            (px - tile_w / 2, py + tile_h / 2)
        ]
        alpha = int(100 + 50 * math.sin(pygame.time.get_ticks() * 0.005))
        cursor_color = self.CRYSTAL_COLORS[self.selected_crystal_type]['main']
        pygame.gfxdraw.aapolygon(self.screen, points, (*cursor_color, 255))
        pygame.gfxdraw.filled_polygon(self.screen, points, (*cursor_color, alpha))

    def _add_particles(self, pos, color):
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': np.array(pos, dtype=float),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'size': random.uniform(2, 5),
                'life': random.uniform(0.5, 1.5),
                'color': color
            })

    def _update_particles(self):
        dt = 1/30.0 # Assume 30fps for particle physics
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel'] * dt
            p['life'] -= dt
            p['size'] = max(0, p['size'] - 2*dt)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "crystals_remaining": sum(self.crystal_inventory),
            "targets_lit": sum(1 for t in self.targets.values() if t['lit']),
            "total_targets": len(self.targets)
        }

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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # --- Manual Play Controls ---
    # Arrow keys: Move
    # Space: Place crystal
    # Left Shift: Cycle crystal type
    # R: Reset
    # Q: Quit
    
    pygame.display.set_caption("Crystal Caverns")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = np.array([0, 0, 0]) # no-op, release, release
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                
                # For manual play, we only take one action per frame
                if not terminated:
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_UP]: action[0] = 1
                    elif keys[pygame.K_DOWN]: action[0] = 2
                    elif keys[pygame.K_LEFT]: action[0] = 3
                    elif keys[pygame.K_RIGHT]: action[0] = 4
                    
                    if keys[pygame.K_SPACE]: action[1] = 1
                    if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

                    obs, reward, terminated, _, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
        
        # Render the environment's observation to the display
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)
        
    env.close()