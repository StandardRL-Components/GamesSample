import gymnasium as gym
import os
import pygame
import math
import numpy as np
from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a puzzle game where the player rotates falling
    geometric shapes to match colors in a circular gravity well. The goal is to
    achieve a high score through combos and chain reactions.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "A puzzle game where the player rotates falling geometric shapes to match colors in a circular gravity well. "
        "Match three or more blocks of the same color to clear them."
    )
    user_guide = "Controls: Use the arrow keys (↑, ↓, →, ←) to rotate the falling piece."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    WELL_CENTER = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50)
    WELL_RADIUS = 160
    BLOCK_SIZE = 20

    # --- Colors ---
    COLOR_BG = (15, 19, 26)
    COLOR_WELL_BG = (25, 30, 40)
    COLOR_WELL_BORDER = (50, 60, 80)
    COLOR_UI_TEXT = (220, 220, 230)
    COLOR_COMBO_TEXT = (255, 200, 0)

    PALETTE = {
        'red': (255, 80, 80),
        'green': (80, 255, 80),
        'blue': (80, 120, 255),
        'yellow': (255, 230, 80),
        'purple': (200, 80, 255),
    }
    COLOR_NAMES = list(PALETTE.keys())

    # --- Shape Definitions (relative block coords from piece center) ---
    SHAPES = {
        'I': [  # 2 rotations
            [(-1, 0), (0, 0), (1, 0), (2, 0)],
            [(0, -1), (0, 0), (0, 1), (0, 2)]
        ],
        'O': [  # 1 rotation
            [(0, 0), (1, 0), (0, 1), (1, 1)]
        ],
        'T': [  # 4 rotations
            [(-1, 0), (0, 0), (1, 0), (0, 1)],
            [(0, -1), (0, 0), (0, 1), (-1, 0)],
            [(-1, 0), (0, 0), (1, 0), (0, -1)],
            [(0, -1), (0, 0), (0, 1), (1, 0)]
        ],
        'L': [  # 4 rotations
            [(-1, -1), (-1, 0), (0, 0), (1, 0)],
            [(-1, 1), (0, 1), (0, 0), (0, -1)],
            [(-1, 0), (0, 0), (1, 0), (1, 1)],
            [(0, 1), (0, 0), (0, -1), (1, -1)]
        ],
        'S': [  # 2 rotations
            [(-1, 1), (0, 1), (0, 0), (1, 0)],
            [(-1, -1), (-1, 0), (0, 0), (0, 1)]
        ]
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 36)
        self.font_combo = pygame.font.Font(None, 48)

        # Initialize state variables
        self.settled_blocks = {}
        self.current_piece = None
        self.next_piece = None
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.initial_fall_speed = 2.0
        self.fall_speed = self.initial_fall_speed
        self.combo_multiplier = 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fall_speed = self.initial_fall_speed
        self.combo_multiplier = 1

        self.settled_blocks = {}
        self.particles = []

        self.current_piece = self._generate_piece()
        self.next_piece = self._generate_piece()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- 1. Handle Action ---
        rotation_action = action[0]
        if rotation_action in [1, 2, 3, 4]:
            num_rotations = len(self.SHAPES[self.current_piece['shape_name']])
            target_rotation = (self.current_piece['rotation'] + 1) % num_rotations
            if rotation_action == 1: # up
                target_rotation = (self.current_piece['rotation'] + 1) % num_rotations
            elif rotation_action == 2: # right
                target_rotation = (self.current_piece['rotation'] + 1) % num_rotations
            elif rotation_action == 3: # down
                target_rotation = (self.current_piece['rotation'] - 1 + num_rotations) % num_rotations
            elif rotation_action == 4: # left
                 target_rotation = (self.current_piece['rotation'] - 1 + num_rotations) % num_rotations


            # Check if rotation is valid before applying
            if self._is_valid_position(self.current_piece, new_rotation=target_rotation):
                self.current_piece['rotation'] = target_rotation

        # --- 2. Update Game Logic ---
        self.current_piece['y'] += self.fall_speed

        if not self._is_valid_position(self.current_piece):
            # Collision detected, move back and settle
            self.current_piece['y'] -= self.fall_speed
            self._settle_piece()

            # --- Chain Reaction Logic ---
            chain_reward, chain_score = self._handle_matches()
            reward += chain_reward
            self.score += chain_score

            # Spawn next piece
            self.current_piece = self.next_piece
            self.next_piece = self._generate_piece()

            # Check for game over
            if not self._is_valid_position(self.current_piece):
                self.game_over = True
                reward = -10.0  # Terminal penalty

        # --- 3. Update Difficulty ---
        if self.steps > 0 and self.steps % 500 == 0:
            self.fall_speed += 0.05

        # --- 4. Update Particles ---
        self._update_particles()

        # --- 5. Check Termination ---
        terminated = self.game_over
        truncated = self.steps >= 5000

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_matches(self):
        total_reward = 0
        total_score = 0
        self.combo_multiplier = 1

        while True:
            matches = self._find_matches()
            if not matches:
                break

            blocks_to_delete = set()
            for group in matches:
                blocks_to_delete.update(group)

            num_cleared = len(blocks_to_delete)

            # Calculate reward and score for this link in the chain
            link_reward = num_cleared * 0.1
            if num_cleared >= 4: link_reward += 1.0
            if num_cleared >= 8: link_reward += 5.0
            link_reward += 10 * self.combo_multiplier

            link_score = num_cleared * 10 * self.combo_multiplier

            total_reward += link_reward
            total_score += link_score

            # Remove matched blocks and create particles
            for grid_pos in blocks_to_delete:
                if grid_pos in self.settled_blocks:
                    color_name = self.settled_blocks[grid_pos]
                    world_pos = self._grid_to_world(grid_pos)
                    self._create_particles(world_pos, self.PALETTE[color_name])
                    del self.settled_blocks[grid_pos]

            self._apply_gravity_to_settled_blocks()

            self.combo_multiplier += 1

        return total_reward, total_score

    def _find_matches(self):
        q = list(self.settled_blocks.keys())
        visited = set()
        all_matches = []

        for grid_pos in q:
            if grid_pos not in visited and grid_pos in self.settled_blocks:
                color = self.settled_blocks[grid_pos]
                component = []
                q_bfs = [grid_pos]
                visited_bfs = {grid_pos}

                while q_bfs:
                    curr = q_bfs.pop(0)
                    component.append(curr)
                    visited.add(curr)

                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        neighbor = (curr[0] + dx, curr[1] + dy)
                        if neighbor in self.settled_blocks and self.settled_blocks[neighbor] == color and neighbor not in visited_bfs:
                            visited_bfs.add(neighbor)
                            q_bfs.append(neighbor)

                if len(component) >= 3:
                    all_matches.append(component)

        return all_matches

    def _apply_gravity_to_settled_blocks(self):
        moved = True
        while moved:
            moved = False
            sorted_blocks = sorted(list(self.settled_blocks.keys()), key=lambda p: p[1], reverse=True)

            for grid_pos in sorted_blocks:
                if grid_pos not in self.settled_blocks: continue
                color = self.settled_blocks[grid_pos]

                below = (grid_pos[0], grid_pos[1] + 1)

                world_pos_below = self._grid_to_world(below)
                dist_from_center = math.hypot(world_pos_below[0] - self.WELL_CENTER[0], world_pos_below[1] - self.WELL_CENTER[1])

                if below not in self.settled_blocks and dist_from_center <= self.WELL_RADIUS - self.BLOCK_SIZE / 2:
                    del self.settled_blocks[grid_pos]
                    self.settled_blocks[below] = color
                    moved = True

    def _settle_piece(self):
        coords = self._get_piece_block_coords(self.current_piece)
        for world_pos in coords:
            grid_pos = self._world_to_grid(world_pos)
            self.settled_blocks[grid_pos] = self.current_piece['color_name']

    def _is_valid_position(self, piece, new_rotation=None):
        rotation = new_rotation if new_rotation is not None else piece['rotation']
        test_piece = {**piece, 'rotation': rotation}

        coords = self._get_piece_block_coords(test_piece)
        for world_pos in coords:
            # Check well boundary only if the block is vertically inside the well's span
            if world_pos[1] > self.WELL_CENTER[1] - self.WELL_RADIUS:
                dist_from_center = math.hypot(world_pos[0] - self.WELL_CENTER[0], world_pos[1] - self.WELL_CENTER[1])
                if dist_from_center > self.WELL_RADIUS - self.BLOCK_SIZE / 2:
                    return False

            # Check collision with settled blocks
            grid_pos = self._world_to_grid(world_pos)
            if grid_pos in self.settled_blocks:
                return False
        return True

    def _generate_piece(self):
        available_shapes = ['I', 'O']
        if self.score >= 1000:
            available_shapes.extend(['T', 'L', 'S'])

        shape_name = self.np_random.choice(available_shapes)
        color_name = self.np_random.choice(self.COLOR_NAMES)

        return {
            'x': self.WELL_CENTER[0],
            'y': self.WELL_CENTER[1] - self.WELL_RADIUS - self.BLOCK_SIZE * 2,
            'rotation': 0,
            'shape_name': shape_name,
            'color_name': color_name,
        }

    def _world_to_grid(self, world_pos):
        return (
            round((world_pos[0] - self.WELL_CENTER[0]) / self.BLOCK_SIZE),
            round((world_pos[1] - self.WELL_CENTER[1]) / self.BLOCK_SIZE)
        )

    def _grid_to_world(self, grid_pos):
        return (
            self.WELL_CENTER[0] + grid_pos[0] * self.BLOCK_SIZE,
            self.WELL_CENTER[1] + grid_pos[1] * self.BLOCK_SIZE
        )

    def _get_piece_block_coords(self, piece):
        shape_rotations = self.SHAPES[piece['shape_name']]
        rotation_idx = piece['rotation'] % len(shape_rotations)
        shape_coords = shape_rotations[rotation_idx]

        world_coords = []
        for sx, sy in shape_coords:
            wx = piece['x'] + sx * self.BLOCK_SIZE
            wy = piece['y'] + sy * self.BLOCK_SIZE
            world_coords.append((wx, wy))
        return world_coords

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
            "combo": self.combo_multiplier,
            "fall_speed": self.fall_speed
        }

    def _render_game(self):
        pygame.gfxdraw.filled_circle(self.screen, self.WELL_CENTER[0], self.WELL_CENTER[1], self.WELL_RADIUS, self.COLOR_WELL_BG)
        pygame.gfxdraw.aacircle(self.screen, self.WELL_CENTER[0], self.WELL_CENTER[1], self.WELL_RADIUS, self.COLOR_WELL_BORDER)
        pygame.gfxdraw.aacircle(self.screen, self.WELL_CENTER[0], self.WELL_CENTER[1], self.WELL_RADIUS - 1, self.COLOR_WELL_BORDER)

        for grid_pos, color_name in self.settled_blocks.items():
            world_pos = self._grid_to_world(grid_pos)
            self._draw_block(world_pos, self.PALETTE[color_name])

        if self.current_piece and not self.game_over:
            coords = self._get_piece_block_coords(self.current_piece)
            color = self.PALETTE[self.current_piece['color_name']]
            for pos in coords:
                self._draw_block(pos, color)

        for p in self.particles:
            p_color = p['color']
            alpha = int(255 * (p['life'] / p['max_life']))
            if alpha > 0:
                s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*p_color, alpha), (p['size'], p['size']), p['size'])
                self.screen.blit(s, (int(p['x']-p['size']), int(p['y']-p['size'])), special_flags=pygame.BLEND_RGBA_ADD)


    def _draw_block(self, center_pos, color):
        x, y = int(center_pos[0]), int(center_pos[1])
        size = self.BLOCK_SIZE
        half = size // 2
        rect = pygame.Rect(x - half, y - half, size, size)

        glow_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*color, 50), (size, size), int(half*1.5))
        self.screen.blit(glow_surf, (x - size, y - size), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.draw.rect(self.screen, color, rect, border_radius=3)

        highlight = tuple(min(255, c + 40) for c in color)
        shadow = tuple(max(0, c - 40) for c in color)
        pygame.draw.line(self.screen, highlight, rect.topleft, rect.topright, 2)
        pygame.draw.line(self.screen, highlight, rect.topleft, rect.bottomleft, 2)
        pygame.draw.line(self.screen, shadow, rect.bottomleft, rect.bottomright, 2)
        pygame.draw.line(self.screen, shadow, rect.topright, rect.bottomright, 2)

    def _render_ui(self):
        score_surf = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (20, 20))

        if self.combo_multiplier > 1:
            combo_text = f"{self.combo_multiplier}x COMBO!"
            combo_surf = self.font_combo.render(combo_text, True, self.COLOR_COMBO_TEXT)
            text_rect = combo_surf.get_rect(center=(self.SCREEN_WIDTH // 2, 50))
            self.screen.blit(combo_surf, text_rect)

        preview_box = pygame.Rect(self.SCREEN_WIDTH - 120, 20, 100, 100)
        pygame.draw.rect(self.screen, self.COLOR_WELL_BG, preview_box, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_WELL_BORDER, preview_box, 2, border_radius=5)

        if self.next_piece:
            preview_piece = {**self.next_piece, 'x': preview_box.centerx, 'y': preview_box.centery, 'rotation': 0}
            coords = self._get_piece_block_coords(preview_piece)
            color = self.PALETTE[preview_piece['color_name']]
            for pos in coords:
                self._draw_block(pos, color)

        if self.game_over:
            over_surf = self.font_combo.render("GAME OVER", True, self.PALETTE['red'])
            over_rect = over_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 50))
            self.screen.blit(over_surf, over_rect)

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'x': pos[0], 'y': pos[1],
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'life': self.np_random.integers(20, 40),
                'max_life': 40,
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Un-dummy the video driver to see the rendering
    os.environ["SDL_VIDEODRIVER"] = "x11"
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Geo Fall")
    clock = pygame.time.Clock()

    terminated = False
    truncated = False
    running = True

    # Mapping keys to actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_RIGHT: 2,
        pygame.K_DOWN: 3,
        pygame.K_LEFT: 4,
    }

    while running:
        action = [0, 0, 0]  # Default no-op action

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Reset
                    obs, info = env.reset()
                    terminated = False
                    truncated = False
                if event.key in key_to_action:
                    action[0] = key_to_action[event.key]

        if not terminated and not truncated:
            obs, reward, terminated, truncated, info = env.step(action)
        else:
            # Allow reset even when game is over
            for event in pygame.event.get(pygame.KEYDOWN):
                 if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                    truncated = False


        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))

        pygame.display.flip()
        clock.tick(GameEnv.metadata["render_fps"])

    env.close()