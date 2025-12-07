
# Generated: 2025-08-28T03:27:49.712029
# Source Brief: brief_04934.md
# Brief Index: 4934

        
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to push all crystals in the chosen direction."
    )

    game_description = (
        "An isometric puzzle game. Push glowing crystals into the designated gaps to complete each level. Plan your moves carefully as you have a limited number of pushes!"
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000

        # Visuals
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_FLOOR = (40, 45, 65)
        self.COLOR_WALL_FACE = (60, 50, 50)
        self.COLOR_WALL_TOP = (85, 75, 75)
        self.COLOR_GAP = (10, 10, 10)
        self.CRYSTAL_COLORS = [
            ((0, 255, 255), (180, 255, 255)),  # Cyan
            ((0, 255, 128), (180, 255, 200)),  # Green
            ((255, 0, 255), (255, 180, 255)),  # Magenta
            ((255, 255, 0), (255, 255, 180)),  # Yellow
        ]
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_UI_BG = (30, 35, 50, 180)

        # Isometric Grid
        self.TILE_WIDTH_HALF = 24
        self.TILE_HEIGHT_HALF = 12
        self.GRID_W, self.GRID_H = 10, 10
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 100
        
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
        self.font_main = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # --- Game State ---
        self.steps = 0
        self.score = 0.0
        self.level = 0
        self.game_over = False
        self.moves_remaining = 0
        self.walls = []
        self.gaps = []
        self.crystals = [] # List of tuples: ((gx, gy), color_index)
        self.particles = []
        
        self.reset()
        
        # --- Validation ---
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.game_over and self._are_all_gaps_filled():
            self.level += 1
        elif self.game_over: # Reset to level 1 on loss
            self.level = 1
        else: # First reset
            self.level = 1

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.particles.clear()
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()
    
    def _generate_level(self):
        self.walls.clear()
        self.gaps.clear()
        self.crystals.clear()

        # Scale difficulty
        self.GRID_W = min(12, 8 + self.level)
        self.GRID_H = min(12, 8 + self.level)
        num_gaps = min(5, 1 + self.level)
        num_crystals = min(6, 2 + self.level)

        # 1. Define walls (boundaries)
        for i in range(-1, self.GRID_W + 1):
            self.walls.append((i, -1))
            self.walls.append((i, self.GRID_H))
        for i in range(self.GRID_H):
            self.walls.append((-1, i))
            self.walls.append((self.GRID_W, i))

        # 2. Generate solvable puzzle
        # Create a set of all possible floor positions
        floor_positions = set((x, y) for x in range(self.GRID_W) for y in range(self.GRID_H))

        # Place gaps
        gap_positions = self.np_random.choice(list(floor_positions), size=num_gaps, replace=False)
        self.gaps = [tuple(pos) for pos in gap_positions]
        
        # Temporarily place crystals in gaps to define a solved state
        available_floor = list(floor_positions - set(self.gaps))
        temp_crystal_pos = list(self.gaps)
        
        # Add extra crystals not in gaps
        if num_crystals > num_gaps:
            extra_crystal_starts = self.np_random.choice(
                available_floor, size=min(len(available_floor), num_crystals - num_gaps), replace=False
            )
            temp_crystal_pos.extend([tuple(p) for p in extra_crystal_starts])

        # Perform random inverse pushes to shuffle the puzzle
        num_shuffles = self.level * 2 + 5
        for _ in range(num_shuffles):
            # Choose a random push direction and apply its inverse
            # 1:up, 2:down, 3:left, 4:right
            inverse_map = {1: 2, 2: 1, 3: 4, 4: 3}
            random_push = self.np_random.integers(1, 5)
            inverse_direction = inverse_map[random_push]
            temp_crystal_pos = self._get_pushed_state(temp_crystal_pos, inverse_direction)

        # Finalize crystal positions with colors
        self.crystals = [(pos, i % len(self.CRYSTAL_COLORS)) for i, pos in enumerate(temp_crystal_pos)]

        # Set moves remaining
        self.moves_remaining = (self.GRID_W + self.GRID_H) + (num_crystals * 5) + (self.level * 5)

    def _get_pushed_state(self, crystal_positions, direction):
        # Helper for level generation, doesn't affect game state
        delta_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        dx, dy = delta_map[direction]

        moved = True
        new_positions = list(crystal_positions)
        
        while moved:
            moved = False
            for i, (cx, cy) in enumerate(new_positions):
                nx, ny = cx + dx, cy + dy
                target_pos = (nx, ny)
                if target_pos not in self.walls and target_pos not in new_positions:
                    new_positions[i] = target_pos
                    moved = True
        return new_positions

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0.0
        
        if movement != 0:
            self.moves_remaining -= 1
            
            # State before move
            pre_move_crystal_pos = [c[0] for c in self.crystals]
            pre_move_filled_gaps = self._get_filled_gaps_count()
            dist_before = self._get_total_crystal_dist_to_gaps()

            # --- Execute Push ---
            self._push_crystals(movement)
            
            # State after move
            dist_after = self._get_total_crystal_dist_to_gaps()
            post_move_filled_gaps = self._get_filled_gaps_count()

            # --- Calculate Reward ---
            # 1. Reward for moving closer to a gap
            reward += (dist_before - dist_after) * 0.1

            # 2. Reward for filling a new gap
            newly_filled = post_move_filled_gaps - pre_move_filled_gaps
            if newly_filled > 0:
                reward += newly_filled * 10.0
                # sfx: gap_filled_sound

        else: # No-op penalty
            reward = -0.01

        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            if self._are_all_gaps_filled():
                # Win condition
                win_bonus = 50.0 + (self.moves_remaining * 0.5)
                reward += win_bonus
                self.score += win_bonus
                # sfx: level_complete_jingle
            else:
                # Loss condition
                reward = -100.0
                self.score -= 100.0
                # sfx: level_fail_sound
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _push_crystals(self, direction):
        # 1:up, 2:down, 3:left, 4:right
        delta_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        dx, dy = delta_map[direction]

        # Sort crystals to push correctly in a chain reaction
        # up (y-): sort by y asc
        # down (y+): sort by y desc
        # left (x-): sort by x asc
        # right (x+): sort by x desc
        sort_key_index = 1 if direction in [1, 2] else 0
        reverse_sort = direction in [2, 4]
        
        sorted_indices = sorted(range(len(self.crystals)), key=lambda k: self.crystals[k][0][sort_key_index], reverse=reverse_sort)
        
        current_crystal_pos = {c[0] for c in self.crystals}

        for i in sorted_indices:
            (cx, cy), color_idx = self.crystals[i]
            original_pos = (cx, cy)
            
            # Slide the crystal
            while True:
                nx, ny = cx + dx, cy + dy
                next_pos = (nx, ny)
                if next_pos in self.walls or next_pos in current_crystal_pos:
                    break # Blocked
                cx, cy = nx, ny
            
            # Update position if moved
            if (cx, cy) != original_pos:
                current_crystal_pos.remove(original_pos)
                current_crystal_pos.add((cx, cy))
                self.crystals[i] = ((cx, cy), color_idx)
                # sfx: crystal_slide_sound
                self._create_particles(original_pos)

    def _create_particles(self, grid_pos):
        sx, sy = self._iso_to_screen(grid_pos[0], grid_pos[1])
        for _ in range(5):
            self.particles.append({
                'pos': [sx, sy + self.TILE_HEIGHT_HALF],
                'vel': [self.np_random.uniform(-1, 1), self.np_random.uniform(-1.5, -0.5)],
                'life': self.np_random.integers(10, 20),
                'color': (100, 100, 120)
            })

    def _get_total_crystal_dist_to_gaps(self):
        if not self.gaps: return 0
        total_dist = 0
        for c_pos, _ in self.crystals:
            if c_pos not in self.gaps:
                min_dist = min(abs(c_pos[0] - g_pos[0]) + abs(c_pos[1] - g_pos[1]) for g_pos in self.gaps)
                total_dist += min_dist
        return total_dist

    def _get_filled_gaps_count(self):
        crystal_positions = {c[0] for c in self.crystals}
        return sum(1 for g in self.gaps if g in crystal_positions)

    def _are_all_gaps_filled(self):
        return self._get_filled_gaps_count() == len(self.gaps)

    def _check_termination(self):
        if self._are_all_gaps_filled():
            self.game_over = True
            return True
        if self.moves_remaining <= 0:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = self.ORIGIN_X + (grid_x - grid_y) * self.TILE_WIDTH_HALF
        screen_y = self.ORIGIN_Y + (grid_x + grid_y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _draw_iso_tile(self, surface, grid_pos, color):
        x, y = self._iso_to_screen(grid_pos[0], grid_pos[1])
        points = [
            (x, y),
            (x + self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF),
            (x, y + self.TILE_HEIGHT_HALF * 2),
            (x - self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF)
        ]
        pygame.draw.polygon(surface, color, points)

    def _draw_iso_cube(self, surface, grid_pos, color_face, color_top, height_mod=1.0):
        x, y = self._iso_to_screen(grid_pos[0], grid_pos[1])
        h = self.TILE_HEIGHT_HALF * 2 * height_mod
        
        # Top face
        top_points = [
            (x, y - h),
            (x + self.TILE_WIDTH_HALF, y - h + self.TILE_HEIGHT_HALF),
            (x, y - h + self.TILE_HEIGHT_HALF * 2),
            (x - self.TILE_WIDTH_HALF, y - h + self.TILE_HEIGHT_HALF)
        ]
        pygame.draw.polygon(surface, color_top, top_points)
        pygame.gfxdraw.aapolygon(surface, top_points, color_top)

        # Left face
        left_points = [
            (x - self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF),
            (x, y + self.TILE_HEIGHT_HALF * 2),
            (x, y - h + self.TILE_HEIGHT_HALF * 2),
            (x - self.TILE_WIDTH_HALF, y - h + self.TILE_HEIGHT_HALF)
        ]
        pygame.draw.polygon(surface, color_face, left_points)

        # Right face
        right_points = [
            (x + self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF),
            (x, y + self.TILE_HEIGHT_HALF * 2),
            (x, y - h + self.TILE_HEIGHT_HALF * 2),
            (x + self.TILE_WIDTH_HALF, y - h + self.TILE_HEIGHT_HALF)
        ]
        pygame.draw.polygon(surface, color_face, right_points)
    
    def _render_game(self):
        # --- Draw Floor and Gaps ---
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                self._draw_iso_tile(self.screen, (x, y), self.COLOR_FLOOR)
        
        for gx, gy in self.gaps:
            self._draw_iso_tile(self.screen, (gx, gy), self.COLOR_GAP)
            x, y = self._iso_to_screen(gx, gy)
            points = [
                (x, y), (x + self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF),
                (x, y + self.TILE_HEIGHT_HALF * 2), (x - self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, (120, 120, 140))

        # --- Sort and Draw Objects ---
        crystal_positions = {c[0] for c in self.crystals}
        
        render_list = []
        for wx, wy in self.walls:
            render_list.append(('wall', (wx, wy)))
        for (cx, cy), color_idx in self.crystals:
            render_list.append(('crystal', (cx, cy), color_idx))

        # Sort by y then x for correct isometric overlap
        render_list.sort(key=lambda item: (item[1][1], item[1][0]))
        
        for item in render_list:
            obj_type = item[0]
            pos = item[1]
            if obj_type == 'wall':
                self._draw_iso_cube(self.screen, pos, self.COLOR_WALL_FACE, self.COLOR_WALL_TOP)
            elif obj_type == 'crystal':
                color_idx = item[2]
                face_color, top_color = self.CRYSTAL_COLORS[color_idx]
                
                # Glow effect
                sx, sy = self._iso_to_screen(pos[0], pos[1])
                glow_color = face_color
                for i in range(5, 0, -1):
                    alpha = 40 - i * 6
                    pygame.gfxdraw.filled_circle(self.screen, sx, sy, 15 + i*2, (*glow_color, alpha))

                self._draw_iso_cube(self.screen, pos, face_color, top_color)

        # --- Update and Draw Particles ---
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                radius = int(p['life'] * 0.2)
                if radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, p['color'])

    def _render_ui(self):
        # Moves remaining
        moves_text = f"Pushes: {self.moves_remaining}"
        moves_surf = self.font_main.render(moves_text, True, self.COLOR_TEXT)
        ui_box = pygame.Rect(5, 5, moves_surf.get_width() + 20, 38)
        pygame.gfxdraw.box(self.screen, ui_box, self.COLOR_UI_BG)
        self.screen.blit(moves_surf, (15, 12))

        # Score
        score_text = f"Score: {int(self.score)}"
        score_surf = self.font_main.render(score_text, True, self.COLOR_TEXT)
        ui_box_score = pygame.Rect(self.WIDTH - score_surf.get_width() - 25, 5, score_surf.get_width() + 20, 38)
        pygame.gfxdraw.box(self.screen, ui_box_score, self.COLOR_UI_BG)
        self.screen.blit(score_surf, (self.WIDTH - score_surf.get_width() - 15, 12))

        # Level
        level_text = f"Level: {self.level}"
        level_surf = self.font_small.render(level_text, True, self.COLOR_TEXT)
        self.screen.blit(level_surf, (15, self.HEIGHT - level_surf.get_height() - 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "moves_remaining": self.moves_remaining,
            "gaps_filled": self._get_filled_gaps_count(),
            "total_gaps": len(self.gaps),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Isometric Crystal Pusher")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    print(GameEnv.game_description)
    print(GameEnv.user_guide)

    while running:
        action = np.array([0, 0, 0]) # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if terminated:
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
                        terminated = False
                    continue

                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                
                # Only step if a valid key was pressed
                if action[0] != 0:
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
                    if terminated:
                        print("Game Over! Press 'R' to restart.")

        # Draw the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit FPS
        
    env.close()