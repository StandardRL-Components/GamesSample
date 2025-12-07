
# Generated: 2025-08-28T02:47:53.515803
# Source Brief: brief_04566.md
# Brief Index: 4566

        
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


class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-1.5, 1.5)
        self.vy = random.uniform(-3, -1)
        self.lifespan = random.randint(20, 40)
        self.color = color

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1  # Gravity
        self.lifespan -= 1

    def draw(self, surface):
        if self.lifespan > 0:
            alpha = max(0, min(255, int(255 * (self.lifespan / 40))))
            # Simple square particle
            rect = pygame.Rect(int(self.x) - 1, int(self.y) - 1, 3, 3)
            temp_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            temp_surf.fill((*self.color, alpha))
            surface.blit(temp_surf, rect.topleft)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrows to move the selected gem. Press space to select/deselect a gem. "
        "If no gem is selected, an action will select the one closest to the center."
    )

    game_description = (
        "An isometric puzzle game. Move all 10 gems to the glowing collection zone at the bottom "
        "of the grid within 50 moves. Plan your moves carefully to solve the puzzle!"
    )

    auto_advance = False

    # --- Colors and Constants ---
    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (60, 65, 80)
    COLOR_COLLECT_ZONE = (70, 90, 120)
    COLOR_TEXT = (230, 230, 230)
    COLOR_SCORE = (255, 215, 0)
    COLOR_MOVES = (173, 216, 230)

    GEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
        (255, 165, 0),   # Orange
        (0, 255, 255),   # Cyan
    ]

    GRID_SIZE = (8, 8)
    NUM_GEMS = 10
    MAX_MOVES = 50
    MAX_STEPS = 500

    TILE_WIDTH = 72
    TILE_HEIGHT = TILE_WIDTH // 2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        
        try:
            self.font_large = pygame.font.Font(None, 36)
            self.font_small = pygame.font.Font(None, 28)
        except IOError:
            # Fallback if default font is not found
            self.font_large = pygame.font.SysFont("sans", 36)
            self.font_small = pygame.font.SysFont("sans", 28)

        self.origin_x = self.screen.get_width() // 2
        self.origin_y = 100
        
        self.gems = []
        self.particles = []
        self.selected_gem_id = None
        self.last_space_press = False
        
        self.reset()

        self.validate_implementation()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        self.gems_collected = 0
        self.selected_gem_id = None
        self.last_space_press = False
        self.particles.clear()
        
        self.collection_row = self.GRID_SIZE[1] - 1
        
        # Procedurally generate gem layout
        self.gems.clear()
        occupied_pos = set()
        
        # Place gems in the upper rows, leaving space
        available_cells = []
        for r in range(self.collection_row - 1):
            for c in range(self.GRID_SIZE[0]):
                available_cells.append((c,r))
        
        if self.NUM_GEMS > len(available_cells):
             raise ValueError("Not enough space on the grid for all gems.")

        gem_positions = self.np_random.choice(len(available_cells), self.NUM_GEMS, replace=False)

        for i in range(self.NUM_GEMS):
            pos = available_cells[gem_positions[i]]
            color = self.np_random.choice(len(self.GEM_COLORS))
            self.gems.append({
                "id": i + 1,
                "pos": pos,
                "color": self.GEM_COLORS[color],
                "collected": False,
                "screen_pos": self._cart_to_iso(*pos)
            })
            occupied_pos.add(pos)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01  # Small penalty for each step to encourage efficiency

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Selection (Spacebar) ---
        space_pressed = space_held and not self.last_space_press
        self.last_space_press = space_held

        if space_pressed:
            if self.selected_gem_id is not None:
                self.selected_gem_id = None
            else:
                self._select_gem_closest_to_center()
        
        # --- Handle Movement (Arrows) ---
        if movement > 0:
            self.moves_left -= 1
            reward -= 0.1 # Cost for making a move

            if self.selected_gem_id is None:
                self._select_gem_closest_to_center()

            if self.selected_gem_id is not None:
                gem = self._get_gem_by_id(self.selected_gem_id)
                if gem:
                    move_deltas = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # Up, Down, Left, Right
                    dx, dy = move_deltas[movement]
                    
                    current_pos = gem['pos']
                    target_pos = (current_pos[0] + dx, current_pos[1] + dy)

                    if self._is_valid_move(target_pos):
                        gem['pos'] = target_pos
                        gem['screen_pos'] = self._cart_to_iso(*target_pos)
                        
                        # Check for collection
                        if target_pos[1] == self.collection_row:
                            self._collect_gem(gem)
                            reward += 10
                        
                        # Deselect after a successful move
                        self.selected_gem_id = None
                    else:
                        # Invalid move penalty
                        reward -= 1.0

        # --- Update game state ---
        self._update_particles()
        
        terminated = self._check_termination()
        if terminated and self.gems_collected == self.NUM_GEMS:
            reward += 100 # Large bonus for winning
            self.score += 100

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _collect_gem(self, gem):
        if not gem['collected']:
            gem['collected'] = True
            self.gems_collected += 1
            self.score += 10
            # Sound effect placeholder: # sfx_collect_gem.play()
            
            # Spawn particles
            sx, sy = gem['screen_pos']
            for _ in range(30):
                self.particles.append(Particle(sx, sy, gem['color']))

    def _check_termination(self):
        if self.gems_collected == self.NUM_GEMS:
            self.game_over = True
            return True
        if self.moves_left <= 0:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_gem_by_id(self, gem_id):
        for gem in self.gems:
            if gem['id'] == gem_id:
                return gem
        return None

    def _select_gem_closest_to_center(self):
        uncollected_gems = [g for g in self.gems if not g['collected']]
        if not uncollected_gems:
            return

        grid_center = (self.GRID_SIZE[0] / 2, self.GRID_SIZE[1] / 2)
        
        closest_gem = min(
            uncollected_gems,
            key=lambda g: math.hypot(g['pos'][0] - grid_center[0], g['pos'][1] - grid_center[1])
        )
        self.selected_gem_id = closest_gem['id']
        # Sound effect placeholder: # sfx_select.play()

    def _is_valid_move(self, target_pos):
        tx, ty = target_pos
        if not (0 <= tx < self.GRID_SIZE[0] and 0 <= ty < self.GRID_SIZE[1]):
            return False # Out of bounds
        
        occupied_positions = {g['pos'] for g in self.gems if not g['collected']}
        if target_pos in occupied_positions:
            return False # Space is occupied
            
        return True

    def _cart_to_iso(self, x, y):
        screen_x = self.origin_x + (x - y) * (self.TILE_WIDTH / 2)
        screen_y = self.origin_y + (x + y) * (self.TILE_HEIGHT / 2)
        return int(screen_x), int(screen_y)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid and collection zone
        for y in range(self.GRID_SIZE[1]):
            for x in range(self.GRID_SIZE[0]):
                screen_pos = self._cart_to_iso(x, y)
                
                # Define the four points of the rhombus tile
                points = [
                    (screen_pos[0], screen_pos[1] - self.TILE_HEIGHT / 2),
                    (screen_pos[0] + self.TILE_WIDTH / 2, screen_pos[1]),
                    (screen_pos[0], screen_pos[1] + self.TILE_HEIGHT / 2),
                    (screen_pos[0] - self.TILE_WIDTH / 2, screen_pos[1]),
                ]
                
                # Draw collection zone tile
                if y == self.collection_row:
                    pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_COLLECT_ZONE)
                
                # Draw grid outline
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

        # Draw gems
        for gem in sorted(self.gems, key=lambda g: g['pos'][1] * 100 + g['pos'][0]):
            if not gem['collected']:
                is_selected = gem['id'] == self.selected_gem_id
                self._draw_gem(gem['screen_pos'], gem['color'], is_selected)
        
        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

    def _draw_gem(self, screen_pos, color, is_selected):
        sx, sy = screen_pos
        
        # Main gem body
        gem_points = [
            (sx, sy - self.TILE_HEIGHT * 0.4),
            (sx + self.TILE_WIDTH * 0.35, sy),
            (sx, sy + self.TILE_HEIGHT * 0.4),
            (sx - self.TILE_WIDTH * 0.35, sy),
        ]
        
        # Highlight/Selection effect
        if is_selected:
            # Pulsing size effect for selection
            pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2 # 0 to 1
            size_mod = 1.0 + pulse * 0.15
            
            # Draw a bright, larger background shape for glow
            glow_color = (255, 255, 255)
            glow_points = [
                (sx, sy - self.TILE_HEIGHT * 0.4 * (size_mod + 0.1)),
                (sx + self.TILE_WIDTH * 0.35 * (size_mod + 0.1), sy),
                (sx, sy + self.TILE_HEIGHT * 0.4 * (size_mod + 0.1)),
                (sx - self.TILE_WIDTH * 0.35 * (size_mod + 0.1), sy),
            ]
            pygame.gfxdraw.filled_polygon(self.screen, glow_points, glow_color)
            pygame.gfxdraw.aapolygon(self.screen, glow_points, glow_color)
        
        pygame.gfxdraw.filled_polygon(self.screen, gem_points, color)
        
        # Add a subtle highlight for 3D effect
        highlight_color = tuple(min(255, c + 80) for c in color)
        highlight_points = [
            gem_points[0],
            (sx, sy - self.TILE_HEIGHT * 0.1),
            gem_points[3],
        ]
        pygame.gfxdraw.filled_polygon(self.screen, highlight_points, highlight_color)
        
        # Outline
        pygame.gfxdraw.aapolygon(self.screen, gem_points, tuple(max(0, c - 50) for c in color))

    def _render_ui(self):
        # Moves Left
        moves_text = f"Moves: {self.moves_left}"
        moves_surf = self.font_large.render(moves_text, True, self.COLOR_MOVES)
        self.screen.blit(moves_surf, (self.screen.get_width() - moves_surf.get_width() - 20, 20))
        
        # Gems Collected
        gems_text = f"Gems: {self.gems_collected} / {self.NUM_GEMS}"
        gems_surf = self.font_large.render(gems_text, True, self.COLOR_SCORE)
        self.screen.blit(gems_surf, (20, 20))

        if self.game_over:
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.gems_collected == self.NUM_GEMS:
                end_text = "PUZZLE SOLVED!"
                end_color = self.COLOR_SCORE
            else:
                end_text = "OUT OF MOVES"
                end_color = self.COLOR_MOVES
            
            end_surf = self.font_large.render(end_text, True, end_color)
            end_rect = end_surf.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(end_surf, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "gems_collected": self.gems_collected,
        }
        
    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a separate display for human play
    pygame.display.set_caption("Gem Shifter")
    human_screen = pygame.display.set_mode((640, 400))
    
    terminated = False
    
    # Game loop for human play
    while not terminated:
        # --- Action mapping for human keyboard input ---
        # Default action is no-op
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        # We only step if an action is taken, mimicking turn-based gameplay
        if movement > 0 or (space == 1 and not env.last_space_press):
            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")
        else:
            # For human play, we need to handle the space release correctly
            env.last_space_press = (space == 1)

        # --- Rendering ---
        # Get the observation from the environment
        frame = env._get_observation()
        # The observation is (H, W, C), but pygame needs (W, H)
        # and the array is transposed. So we need to transpose it back.
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        human_screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        
        # In human mode, we can add a small delay to make it playable
        env.clock.tick(30)
        
    env.close()
    print("Game Over.")