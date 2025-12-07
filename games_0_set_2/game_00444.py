
# Generated: 2025-08-27T13:39:59.809096
# Source Brief: brief_00444.md
# Brief Index: 444

        
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
        "Controls: Space/Shift to cycle selected block. ↑↓←→ to push the selected block."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Push pixelated blocks to their target locations in this isometric puzzle game. Plan your moves carefully to solve the level."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 9
    GRID_HEIGHT = 7
    TILE_WIDTH_ISO = 48
    TILE_HEIGHT_ISO = 24
    MAX_STEPS = 250

    # Colors
    COLOR_BG = (30, 32, 36)
    COLOR_GRID = (50, 52, 58)
    COLOR_TARGET_GOLD = (255, 215, 0)
    COLOR_UI_TEXT = (220, 220, 220)
    BLOCK_COLORS = [
        (255, 87, 87),   # Red
        (87, 155, 255),  # Blue
        (87, 255, 155),  # Green
        (255, 155, 87),  # Orange
        (155, 87, 255),  # Purple
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Arial", 18, bold=True)
        self.font_msg = pygame.font.SysFont("Arial", 32, bold=True)
        
        # Centering offset for the grid
        self.origin_x = self.SCREEN_WIDTH // 2
        self.origin_y = 100

        # Initialize state variables
        self.blocks = []
        self.targets = []
        self.particles = []
        self.selected_block_idx = 0
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win_message = ""
        
        # This will be set in reset()
        self.np_random = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win_message = ""
        self.selected_block_idx = 0
        self.particles.clear()

        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.blocks.clear()
        self.targets.clear()
        
        num_blocks = self.np_random.integers(3, 6)
        
        occupied_coords = set()
        
        # Place targets
        for i in range(num_blocks):
            while True:
                pos = (
                    self.np_random.integers(0, self.GRID_WIDTH),
                    self.np_random.integers(0, self.GRID_HEIGHT)
                )
                if pos not in occupied_coords:
                    occupied_coords.add(pos)
                    self.targets.append({
                        "pos": pos,
                        "color": self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
                    })
                    break
        
        # Place blocks
        for i in range(num_blocks):
            while True:
                pos = (
                    self.np_random.integers(0, self.GRID_WIDTH),
                    self.np_random.integers(0, self.GRID_HEIGHT)
                )
                if pos not in occupied_coords:
                    occupied_coords.add(pos)
                    self.blocks.append({
                        "pos": pos,
                        "color": self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
                    })
                    break

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        reward = -0.1  # Cost per step
        
        # 1. Handle selection change
        if space_held and not shift_held:
            self.selected_block_idx = (self.selected_block_idx + 1) % len(self.blocks)
        elif shift_held and not space_held:
            self.selected_block_idx = (self.selected_block_idx - 1 + len(self.blocks)) % len(self.blocks)
        
        # 2. Handle push action
        if movement > 0:
            push_reward = self._handle_push(movement)
            reward += push_reward
        
        self.steps += 1
        self.score += reward
        
        self._update_particles()
        
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            is_win = self._check_win_condition()
            if is_win:
                terminal_reward = 100.0
                self.win_message = "LEVEL COMPLETE!"
            else:
                terminal_reward = -100.0
                self.win_message = "STUCK!" if self.steps < self.MAX_STEPS else "TIME UP!"
            self.score += terminal_reward
            reward += terminal_reward
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_push(self, direction_idx):
        # 1=up, 2=down, 3=left, 4=right
        direction_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        dx, dy = direction_map[direction_idx]

        block_positions = {b['pos']: b for b in self.blocks}
        
        line_of_blocks = []
        current_pos = self.blocks[self.selected_block_idx]['pos']
        
        # Trace the line of blocks to be pushed
        while current_pos in block_positions:
            line_of_blocks.append(block_positions[current_pos])
            current_pos = (current_pos[0] + dx, current_pos[1] + dy)

        # Check if the move is valid (the space at the end of the line is free)
        final_pos = current_pos
        if not (0 <= final_pos[0] < self.GRID_WIDTH and 0 <= final_pos[1] < self.GRID_HEIGHT):
            return 0.0 # Pushed into a wall

        # Calculate reward change
        on_target_before = self._count_on_target(line_of_blocks)
        
        # Move the blocks in reverse order
        for block in reversed(line_of_blocks):
            original_pos = block['pos']
            block['pos'] = (block['pos'][0] + dx, block['pos'][1] + dy)
            self._add_particles(original_pos, block['color'])
            # Sound effect placeholder: # sfx_push_block()
        
        on_target_after = self._count_on_target(line_of_blocks)
        
        return float(on_target_after - on_target_before)

    def _count_on_target(self, block_list):
        count = 0
        target_positions = {t['pos']: t['color'] for t in self.targets}
        for block in block_list:
            if block['pos'] in target_positions and target_positions[block['pos']] == block['color']:
                count += 1
        return count

    def _check_termination(self):
        if self._check_win_condition():
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        if not self._any_valid_moves():
            return True
        return False

    def _check_win_condition(self):
        return self._count_on_target(self.blocks) == len(self.blocks)

    def _any_valid_moves(self):
        for i in range(len(self.blocks)):
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                if self._is_push_valid(i, dx, dy):
                    return True
        return False

    def _is_push_valid(self, block_idx, dx, dy):
        block_positions = {b['pos'] for b in self.blocks}
        current_pos = self.blocks[block_idx]['pos']
        
        while current_pos in block_positions:
            current_pos = (current_pos[0] + dx, current_pos[1] + dy)
        
        return 0 <= current_pos[0] < self.GRID_WIDTH and 0 <= current_pos[1] < self.GRID_HEIGHT

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_HEIGHT + 1):
            start = self._iso_to_screen(0, y)
            end = self._iso_to_screen(self.GRID_WIDTH, y)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for x in range(self.GRID_WIDTH + 1):
            start = self._iso_to_screen(x, 0)
            end = self._iso_to_screen(x, self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

        # Collect all items to draw and sort by depth
        drawables = []
        for target in self.targets:
            drawables.append(('target', target))
        for block in self.blocks:
            drawables.append(('block', block))

        drawables.sort(key=lambda item: item[1]['pos'][0] + item[1]['pos'][1])

        for item_type, item_data in drawables:
            if item_type == 'target':
                self._draw_iso_target(item_data['pos'], item_data['color'])
            elif item_type == 'block':
                self._draw_iso_cube(item_data['pos'], item_data['color'])
        
        self._render_particles()
        self._render_selection_cursor()

    def _iso_to_screen(self, x, y):
        screen_x = self.origin_x + (x - y) * (self.TILE_WIDTH_ISO / 2)
        screen_y = self.origin_y + (x + y) * (self.TILE_HEIGHT_ISO / 2)
        return int(screen_x), int(screen_y)

    def _draw_iso_cube(self, pos, color):
        x, y = pos
        sx, sy = self._iso_to_screen(x, y)
        
        height = 30
        
        top_face = [
            (sx, sy - height),
            (sx + self.TILE_WIDTH_ISO / 2, sy - self.TILE_HEIGHT_ISO / 2 - height),
            (sx, sy - self.TILE_HEIGHT_ISO - height),
            (sx - self.TILE_WIDTH_ISO / 2, sy - self.TILE_HEIGHT_ISO / 2 - height)
        ]
        
        left_face = [
            (sx, sy), (sx, sy - height),
            (sx - self.TILE_WIDTH_ISO / 2, sy - self.TILE_HEIGHT_ISO / 2 - height),
            (sx - self.TILE_WIDTH_ISO / 2, sy - self.TILE_HEIGHT_ISO / 2)
        ]

        right_face = [
            (sx, sy), (sx, sy - height),
            (sx + self.TILE_WIDTH_ISO / 2, sy - self.TILE_HEIGHT_ISO / 2 - height),
            (sx + self.TILE_WIDTH_ISO / 2, sy - self.TILE_HEIGHT_ISO / 2)
        ]
        
        # Shading
        light_color = tuple(min(255, c + 30) for c in color)
        dark_color = tuple(max(0, c - 30) for c in color)
        
        pygame.draw.polygon(self.screen, light_color, top_face)
        pygame.draw.polygon(self.screen, color, left_face)
        pygame.draw.polygon(self.screen, dark_color, right_face)
        
        # Outline for clarity
        pygame.draw.aalines(self.screen, self.COLOR_BG, True, top_face)
        pygame.draw.aalines(self.screen, self.COLOR_BG, True, left_face)
        pygame.draw.aalines(self.screen, self.COLOR_BG, True, right_face)

    def _draw_iso_target(self, pos, color):
        x, y = pos
        sx, sy = self._iso_to_screen(x, y)
        
        points = [
            (sx, sy),
            (sx + self.TILE_WIDTH_ISO / 2, sy - self.TILE_HEIGHT_ISO / 2),
            (sx, sy - self.TILE_HEIGHT_ISO),
            (sx - self.TILE_WIDTH_ISO / 2, sy - self.TILE_HEIGHT_ISO / 2)
        ]
        
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_TARGET_GOLD)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_TARGET_GOLD)
        
        # Inner shape to match block color
        inner_points = [
            (sx, sy - 4),
            (sx + self.TILE_WIDTH_ISO / 2 - 4, sy - self.TILE_HEIGHT_ISO / 2 - 2),
            (sx, sy - self.TILE_HEIGHT_ISO + 4),
            (sx - self.TILE_WIDTH_ISO / 2 + 4, sy - self.TILE_HEIGHT_ISO / 2 - 2)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, inner_points, color)
        pygame.gfxdraw.aapolygon(self.screen, inner_points, color)

    def _render_selection_cursor(self):
        if self.game_over or not self.blocks:
            return
        
        block = self.blocks[self.selected_block_idx]
        sx, sy = self._iso_to_screen(block['pos'][0], block['pos'][1])
        
        cursor_height = 45 + 5 * math.sin(self.steps * 0.2)
        
        points = [
            (sx - 10, sy - cursor_height - 10),
            (sx, sy - cursor_height),
            (sx + 10, sy - cursor_height - 10)
        ]
        
        pygame.draw.polygon(self.screen, (255, 255, 255), points)
        pygame.draw.aalines(self.screen, (0,0,0), True, points)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.font_ui.render(f"MOVES: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (10, 30))

        if self.game_over:
            msg_surf = self.font_msg.render(self.win_message, True, (255, 255, 255))
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 150))
            
            bg_rect = msg_rect.inflate(20, 10)
            pygame.draw.rect(self.screen, self.COLOR_BG, bg_rect, border_radius=5)
            pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, bg_rect, width=2, border_radius=5)
            
            self.screen.blit(msg_surf, msg_rect)

    def _add_particles(self, grid_pos, color):
        sx, sy = self._iso_to_screen(grid_pos[0], grid_pos[1])
        for _ in range(15):
            particle = {
                'x': sx,
                'y': sy - self.TILE_HEIGHT_ISO,
                'vx': self.np_random.uniform(-1.5, 1.5),
                'vy': self.np_random.uniform(-2.5, 0.5),
                'life': self.np_random.integers(20, 40),
                'color': color,
                'size': self.np_random.uniform(2, 5)
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # Gravity
            p['life'] -= 1
            p['size'] -= 0.1
        self.particles = [p for p in self.particles if p['life'] > 0 and p['size'] > 0]
    
    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 40.0))))
            color_with_alpha = p['color'] + (alpha,)
            
            # Create a temporary surface for the particle to handle alpha
            particle_surf = pygame.Surface((int(p['size']*2), int(p['size']*2)), pygame.SRCALPHA)
            pygame.draw.rect(particle_surf, color_with_alpha, particle_surf.get_rect())
            self.screen.blit(particle_surf, (int(p['x']), int(p['y'])))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "blocks_on_target": self._count_on_target(self.blocks),
            "total_blocks": len(self.blocks),
            "is_win": self._check_win_condition() if self.game_over else False
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
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
    pygame.display.set_caption("Isometric Block Pusher")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = np.array([0, 0, 0])  # no-op, released, released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        # In manual play, we only step when an action is taken
        if np.any(action > 0):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
            if terminated:
                print("Game Over! Resetting in 3 seconds...")
                
                # Display final frame
                frame = np.transpose(obs, (1, 0, 2))
                surf = pygame.surfarray.make_surface(frame)
                screen.blit(surf, (0, 0))
                pygame.display.flip()
                
                pygame.time.wait(3000)
                obs, info = env.reset()
        
        # Get the latest observation for rendering
        current_obs = env._get_observation()
        frame = np.transpose(current_obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)
        
    env.close()