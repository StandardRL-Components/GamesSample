
# Generated: 2025-08-27T18:38:11.820190
# Source Brief: brief_01888.md
# Brief Index: 1888

        
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
    """
    A minimalist puzzle game where the player connects nodes of the same color on a grid.
    The goal is to clear the entire board before running out of moves.
    This environment is designed with a focus on visual clarity and satisfying feedback.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move selector. Space to select a node, "
        "then space on another node to connect. Shift to deselect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Connect same-colored nodes to clear the board. Each connection attempt costs a move. "
        "Clearing adjacent nodes of the same color creates a chain reaction."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        """Initializes the game environment."""
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 16, 10
        self.CELL_W = self.WIDTH // self.GRID_W
        self.CELL_H = self.HEIGHT // self.GRID_H
        self.NUM_PAIRS = self.np_random.integers(8, 13)
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_SELECTOR = (255, 255, 255)
        self.COLOR_FAIL = (255, 80, 80)
        self.NODE_COLORS = [
            (52, 152, 219),  # Blue
            (231, 76, 60),   # Red
            (46, 204, 113),  # Green
            (241, 196, 15),  # Yellow
            (155, 89, 182),  # Purple
            (230, 126, 34),  # Orange
        ]

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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 50)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = 0
        self.nodes = []
        self.node_pos_map = {}
        self.selector_pos = (0, 0)
        self.selected_node_idx = None
        self.prev_space_held = False
        self.prev_shift_held = False
        self.visual_effects = []
        
        self.reset()
        # self.validate_implementation()


    def reset(self, seed=None, options=None):
        """Resets the environment to its initial state."""
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.selector_pos = (self.GRID_W // 2, self.GRID_H // 2)
        self.selected_node_idx = None
        self.prev_space_held = False
        self.prev_shift_held = False
        self.visual_effects = []
        
        self._generate_puzzle()

        return self._get_observation(), self._get_info()

    def step(self, action):
        """Processes an action and updates the game state."""
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        # 1. Handle Selector Movement
        if movement == 1: self.selector_pos = (self.selector_pos[0], max(0, self.selector_pos[1] - 1))
        elif movement == 2: self.selector_pos = (self.selector_pos[0], min(self.GRID_H - 1, self.selector_pos[1] + 1))
        elif movement == 3: self.selector_pos = (max(0, self.selector_pos[0] - 1), self.selector_pos[1])
        elif movement == 4: self.selector_pos = (min(self.GRID_W - 1, self.selector_pos[0] + 1), self.selector_pos[1])

        # 2. Handle Deselection
        if shift_pressed and self.selected_node_idx is not None:
            self.selected_node_idx = None
            # sound: deselect_sound

        # 3. Handle Selection / Connection
        if space_pressed:
            target_idx = self.node_pos_map.get(self.selector_pos)

            if target_idx is not None:
                if self.selected_node_idx is None:
                    # Select the first node
                    self.selected_node_idx = target_idx
                    # sound: select_node_sound
                elif target_idx != self.selected_node_idx:
                    # Attempt to connect to a second node
                    reward += self._attempt_connection(target_idx)
                    self.moves_remaining -= 1

        self.steps += 1
        self._update_visual_effects()
        
        # 4. Check for Termination Conditions
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if len(self.nodes) == 0:
                reward += 100 # Win bonus
                self.score += 1000
                self.visual_effects.append(self._create_effect('win_text', lifetime=120))
            else:
                reward -= 100 # Loss penalty
                self.visual_effects.append(self._create_effect('lose_text', lifetime=120))
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _attempt_connection(self, target_idx):
        """Handles the logic for connecting two nodes."""
        reward = 0
        node1 = self.nodes[self.selected_node_idx]
        node2 = self.nodes[target_idx]

        if node1['color'] == node2['color']:
            # --- Successful Connection ---
            # sound: connect_success_sound
            reward += 1.0
            
            # Find the entire group to clear via flood fill
            nodes_to_clear_indices = self._find_connected_component(self.selected_node_idx, target_idx)
            
            if len(nodes_to_clear_indices) > 2:
                reward += 5.0 # Bonus for clearing a group
                # sound: group_clear_sound
            
            # Create visual effects and update score
            cleared_nodes_data = []
            for idx in sorted(list(nodes_to_clear_indices), reverse=True):
                cleared_node = self.nodes.pop(idx)
                cleared_nodes_data.append(cleared_node)
                self.score += 10 * len(nodes_to_clear_indices) # Score for each cleared node
                self.visual_effects.append(self._create_effect('burst', lifetime=20, pos=cleared_node['pos'], color=cleared_node['color']))

            self.visual_effects.append(self._create_effect('line', lifetime=25, start_pos=node1['pos'], end_pos=node2['pos'], color=node1['color']))
            
            # Rebuild the position map for faster lookups
            self._update_node_map()
        else:
            # --- Failed Connection ---
            # sound: connect_fail_sound
            reward -= 0.1
            self.visual_effects.append(self._create_effect('fail_indicator', lifetime=30, pos1=node1['pos'], pos2=node2['pos'], color=self.COLOR_FAIL))
        
        self.selected_node_idx = None
        return reward

    def _get_observation(self):
        """Renders the current game state to a numpy array."""
        self.screen.fill(self.COLOR_BG)
        self._draw_grid()
        self._draw_nodes()
        self._draw_selector()
        self._draw_visual_effects()
        self._draw_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        """Returns a dictionary with auxiliary game information."""
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "nodes_left": len(self.nodes),
        }

    # --- Helper and Drawing Methods ---

    def _generate_puzzle(self):
        """Creates a new random puzzle, ensuring it's solvable."""
        self.NUM_PAIRS = self.np_random.integers(8, 13)
        self.moves_remaining = self.NUM_PAIRS * 2
        all_coords = [(x, y) for x in range(self.GRID_W) for y in range(self.GRID_H)]
        self.np_random.shuffle(all_coords)

        self.nodes = []
        for i in range(self.NUM_PAIRS):
            pos1 = all_coords.pop()
            pos2 = all_coords.pop()
            color = self.NODE_COLORS[self.np_random.integers(len(self.NODE_COLORS))]
            
            self.nodes.append({'pos': pos1, 'color': color, 'id': i * 2, 'radius_scale': 1.0})
            self.nodes.append({'pos': pos2, 'color': color, 'id': i * 2 + 1, 'radius_scale': 1.0})
        
        self._update_node_map()

    def _update_node_map(self):
        """Updates the mapping from grid positions to node indices."""
        self.node_pos_map = {node['pos']: i for i, node in enumerate(self.nodes)}

    def _check_termination(self):
        """Checks if the episode should end."""
        win = len(self.nodes) == 0
        loss_moves = self.moves_remaining <= 0
        timeout = self.steps >= self.MAX_STEPS
        return win or loss_moves or timeout

    def _find_connected_component(self, start_idx, end_idx):
        """Finds all nodes of the same color connected by adjacency (flood fill)."""
        start_node = self.nodes[start_idx]
        color = start_node['color']
        
        component_indices = set()
        to_visit = [start_node['pos'], self.nodes[end_idx]['pos']]
        visited_pos = set(to_visit)

        while to_visit:
            x, y = to_visit.pop(0)
            
            # Check if current pos has a node and add it
            node_idx = self.node_pos_map.get((x, y))
            if node_idx is not None and self.nodes[node_idx]['color'] == color:
                component_indices.add(node_idx)

                # Check neighbors
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H and (nx, ny) not in visited_pos:
                        neighbor_idx = self.node_pos_map.get((nx, ny))
                        if neighbor_idx is not None and self.nodes[neighbor_idx]['color'] == color:
                            to_visit.append((nx, ny))
                        visited_pos.add((nx, ny))
        return component_indices

    def _grid_to_pixel(self, grid_pos):
        """Converts grid coordinates to pixel coordinates."""
        x = grid_pos[0] * self.CELL_W + self.CELL_W // 2
        y = grid_pos[1] * self.CELL_H + self.CELL_H // 2
        return int(x), int(y)

    def _draw_grid(self):
        """Draws the background grid."""
        for x in range(0, self.WIDTH, self.CELL_W):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_H):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _draw_nodes(self):
        """Draws all nodes on the grid."""
        radius = min(self.CELL_W, self.CELL_H) // 2 - 4
        for i, node in enumerate(self.nodes):
            px, py = self._grid_to_pixel(node['pos'])
            current_radius = int(radius * node['radius_scale'])
            if current_radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, px, py, current_radius, node['color'])
                pygame.gfxdraw.aacircle(self.screen, px, py, current_radius, node['color'])

            if i == self.selected_node_idx:
                pulse = (math.sin(self.steps * 0.2) + 1) / 2
                halo_radius = current_radius + 2 + int(pulse * 3)
                halo_color = (*node['color'], int(100 + pulse * 50)) # RGBA
                s = pygame.Surface((halo_radius*2, halo_radius*2), pygame.SRCALPHA)
                pygame.draw.circle(s, halo_color, (halo_radius, halo_radius), halo_radius)
                self.screen.blit(s, (px - halo_radius, py - halo_radius))


    def _draw_selector(self):
        """Draws the player's selector."""
        pulse = (math.sin(self.steps * 0.25) + 1) / 2
        alpha = int(100 + pulse * 100)
        color = (*self.COLOR_SELECTOR, alpha)
        
        px, py = self.selector_pos
        rect = pygame.Rect(px * self.CELL_W, py * self.CELL_H, self.CELL_W, self.CELL_H)
        
        shape_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, color, shape_surf.get_rect(), border_radius=5)
        self.screen.blit(shape_surf, rect.topleft)

    def _draw_ui(self):
        """Draws the user interface (score, moves)."""
        moves_text = self.font_ui.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (10, 10))

        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)

    def _update_visual_effects(self):
        """Updates and removes expired visual effects."""
        self.visual_effects = [e for e in self.visual_effects if e['update'](e)]

    def _create_effect(self, type, lifetime, **kwargs):
        """Factory for creating visual effect dictionaries."""
        effect = {
            'type': type,
            'lifetime': lifetime,
            'max_lifetime': lifetime,
            'data': kwargs,
            'update': lambda e: e.update({'lifetime': e['lifetime'] - 1}) or e['lifetime'] > 0
        }
        return effect

    def _draw_visual_effects(self):
        """Renders all active visual effects."""
        for effect in self.visual_effects:
            progress = 1.0 - (effect['lifetime'] / effect['max_lifetime'])
            data = effect['data']
            
            if effect['type'] == 'line':
                start_px = self._grid_to_pixel(data['start_pos'])
                end_px = self._grid_to_pixel(data['end_pos'])
                alpha = int(255 * (1 - progress))
                pygame.draw.aaline(self.screen, (*data['color'], alpha), start_px, end_px, 2)

            elif effect['type'] == 'burst':
                px, py = self._grid_to_pixel(data['pos'])
                num_particles = 8
                for i in range(num_particles):
                    angle = (i / num_particles) * 2 * math.pi + self.steps * 0.1
                    dist = progress * self.CELL_W * 1.5
                    p_x = px + int(math.cos(angle) * dist)
                    p_y = py + int(math.sin(angle) * dist)
                    size = int(5 * (1 - progress))
                    if size > 0:
                        pygame.draw.circle(self.screen, data['color'], (p_x, p_y), size)
            
            elif effect['type'] == 'fail_indicator':
                p1 = self._grid_to_pixel(data['pos1'])
                p2 = self._grid_to_pixel(data['pos2'])
                alpha = int(255 * math.sin(progress * math.pi)) # Fade in and out
                pygame.draw.line(self.screen, (*data['color'], alpha), p1, p2, 3)

            elif effect['type'] == 'win_text':
                alpha = int(255 * min(1, progress * 4))
                text = self.font_title.render("PUZZLE CLEARED", True, (*self.NODE_COLORS[2], alpha))
                rect = text.get_rect(center=(self.WIDTH//2, self.HEIGHT//2))
                self.screen.blit(text, rect)

            elif effect['type'] == 'lose_text':
                alpha = int(255 * min(1, progress * 4))
                text = self.font_title.render("OUT OF MOVES", True, (*self.COLOR_FAIL, alpha))
                rect = text.get_rect(center=(self.WIDTH//2, self.HEIGHT//2))
                self.screen.blit(text, rect)

    def close(self):
        """Cleans up Pygame resources."""
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # For a turn-based game, we need a different kind of loop
    # We will display the screen and wait for a key press
    
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Connect Nodes")
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op
    
    running = True
    while running:
        # Map Pygame events to a Gym action
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = np.array([movement, space, shift])
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_remaining']}")
            
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print("Game Over!")
            print(f"Final Info: {info}")
            pygame.time.wait(2000) # Wait 2 seconds
            obs, info = env.reset()

        env.clock.tick(30) # Limit to 30 FPS

    env.close()