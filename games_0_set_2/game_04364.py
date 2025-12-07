
# Generated: 2025-08-28T02:11:30.261865
# Source Brief: brief_04364.md
# Brief Index: 4364

        
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
    An isometric puzzle game where the player pushes crystals into a collection zone
    while avoiding moving traps. The game is turn-based and emphasizes strategic
    planning.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrow keys to push all crystals in a direction. "
        "The goal is to move them to the glowing collection zone."
    )

    # Short, user-facing description of the game
    game_description = (
        "A turn-based isometric puzzle. Push all crystals in the same direction each turn. "
        "Guide them to the collection zone, but beware of moving traps that destroy them."
    )

    # Frames only advance when an action is received
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 14
    GRID_HEIGHT = 14
    TILE_WIDTH_HALF = 20
    TILE_HEIGHT_HALF = 10

    NUM_CRYSTALS = 10
    NUM_TRAPS = 3
    MAX_TRAPS_HIT = 3
    MAX_STEPS = 1000

    COLOR_BG = (15, 18, 32)
    COLOR_GRID = (40, 45, 60)
    COLOR_PLAYER_MARKER = (255, 255, 0)
    COLOR_COLLECT_ZONE = (0, 255, 255)
    COLOR_TRAP = (255, 50, 50)
    CRYSTAL_COLORS = [
        (100, 150, 255), (255, 100, 150), (150, 255, 100),
        (255, 150, 255), (150, 255, 255), (255, 255, 150)
    ]
    COLOR_TEXT = (220, 220, 230)
    COLOR_TEXT_SHADOW = (20, 20, 25)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_title = pygame.font.SysFont("monospace", 36, bold=True)
        
        # Calculate grid offset for centering
        self.origin_x = self.SCREEN_WIDTH // 2
        self.origin_y = self.SCREEN_HEIGHT // 2 - (self.GRID_HEIGHT * self.TILE_HEIGHT_HALF // 2) + 50

        # Initialize state variables
        self.reset()
        
        # Self-check to ensure API compliance
        # self.validate_implementation()


    def _grid_to_iso(self, x, y):
        """Converts grid coordinates to isometric screen coordinates."""
        screen_x = self.origin_x + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.origin_y + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _draw_iso_tile(self, surface, x, y, color, outline_color=None, width=1):
        """Draws a single filled isometric tile."""
        points = [
            self._grid_to_iso(x, y),
            self._grid_to_iso(x + 1, y),
            self._grid_to_iso(x + 1, y + 1),
            self._grid_to_iso(x, y + 1),
        ]
        pygame.gfxdraw.filled_polygon(surface, points, color)
        if outline_color:
            pygame.gfxdraw.aapolygon(surface, points, outline_color)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.crystals_collected = 0
        self.traps_triggered = 0
        self.game_over = False
        self.last_action_reward = 0
        self.last_event = ""
        
        self.collection_zone = (self.GRID_WIDTH - 2, self.GRID_HEIGHT // 2)

        # Generate level
        available_coords = set()
        for x in range(1, self.GRID_WIDTH - 1):
            for y in range(1, self.GRID_HEIGHT - 1):
                available_coords.add((x, y))
        
        available_coords.discard(self.collection_zone)

        # Place crystals
        self.crystals = []
        crystal_coords = self.np_random.choice(list(available_coords), self.NUM_CRYSTALS, replace=False)
        for i, coord_tuple in enumerate(crystal_coords):
            coord = tuple(coord_tuple)
            self.crystals.append({
                "pos": coord,
                "color": random.choice(self.CRYSTAL_COLORS),
                "id": i
            })
            available_coords.remove(coord)

        # Place traps
        self.traps = []
        trap_coords = self.np_random.choice(list(available_coords), self.NUM_TRAPS, replace=False)
        for i, coord_tuple in enumerate(trap_coords):
            coord = tuple(coord_tuple)
            patterns = [
                [(0, -1), (1, 0), (0, 1), (-1, 0)], # Clockwise
                [(0, 1), (1, 0), (0, -1), (-1, 0)], # Counter-clockwise
                [(1, 0), (-1, 0), (1, 0), (-1, 0)], # Horizontal
                [(0, 1), (0, -1), (0, 1), (0, -1)], # Vertical
            ]
            self.traps.append({
                "pos": coord,
                "pattern": random.choice(patterns),
                "pattern_idx": self.np_random.integers(0, 4)
            })
            available_coords.remove(coord)

        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        self.last_event = ""

        if movement != 0: # No-op action
            self.steps += 1
            
            # --- 1. Calculate pre-move state for reward shaping ---
            crystal_positions = np.array([c['pos'] for c in self.crystals])
            trap_positions = np.array([t['pos'] for t in self.traps])
            collection_zone_np = np.array(self.collection_zone)

            # Distance from crystals to collection zone
            if len(crystal_positions) > 0:
                old_dist_to_collection = np.sum(np.abs(crystal_positions - collection_zone_np))
            else:
                old_dist_to_collection = 0
            
            # --- 2. Move Crystals ---
            move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # up, down, left, right
            dx, dy = move_map[movement]
            
            # Sort crystals to handle pushes correctly
            self.crystals.sort(key=lambda c: c['pos'][0] * dx + c['pos'][1] * dy, reverse=True)
            
            occupied_tiles = {c['pos'] for c in self.crystals}
            for crystal in self.crystals:
                current_pos = crystal['pos']
                target_pos = (current_pos[0] + dx, current_pos[1] + dy)

                # Check boundaries and other crystals
                if not (0 < target_pos[0] < self.GRID_WIDTH - 1 and 0 < target_pos[1] < self.GRID_HEIGHT - 1):
                    continue # Hit wall
                if target_pos in occupied_tiles:
                    continue # Hit another crystal
                
                # Update position
                crystal['pos'] = target_pos
                occupied_tiles.remove(current_pos)
                occupied_tiles.add(target_pos)

            # --- 3. Check for Interactions (Collections/Traps) ---
            crystals_to_remove = []
            current_trap_positions = {t['pos'] for t in self.traps}

            for crystal in self.crystals:
                if crystal['pos'] == self.collection_zone:
                    # Crystal collected
                    reward += 10.0
                    self.score += 10
                    self.crystals_collected += 1
                    crystals_to_remove.append(crystal)
                    self.last_event = f"+10 CRYSTAL ({self.crystals_collected}/{self.NUM_CRYSTALS})"
                    # sound: crystal_collect.wav
                    for _ in range(30):
                        self.particles.append(self._create_particle(crystal['pos'], self.COLOR_COLLECT_ZONE, 30))
                elif crystal['pos'] in current_trap_positions:
                    # Crystal hit a trap
                    reward -= 30.0
                    self.score -= 30
                    self.traps_triggered += 1
                    crystals_to_remove.append(crystal)
                    self.last_event = f"-30 TRAP! ({self.traps_triggered}/{self.MAX_TRAPS_HIT})"
                    # sound: explosion.wav
                    for _ in range(50):
                        self.particles.append(self._create_particle(crystal['pos'], self.COLOR_TRAP, 40))

            self.crystals = [c for c in self.crystals if c not in crystals_to_remove]
            
            # --- 4. Move Traps ---
            for trap in self.traps:
                move = trap['pattern'][trap['pattern_idx']]
                target_pos = (trap['pos'][0] + move[0], trap['pos'][1] + move[1])
                
                if 0 < target_pos[0] < self.GRID_WIDTH - 1 and 0 < target_pos[1] < self.GRID_HEIGHT - 1:
                    trap['pos'] = target_pos
                
                trap['pattern_idx'] = (trap['pattern_idx'] + 1) % len(trap['pattern'])

            # --- 5. Calculate post-move reward shaping ---
            crystal_positions = np.array([c['pos'] for c in self.crystals])
            if len(crystal_positions) > 0:
                new_dist_to_collection = np.sum(np.abs(crystal_positions - collection_zone_np))
                # Reward for reducing distance to collection zone
                reward += (old_dist_to_collection - new_dist_to_collection) * 0.1
            else:
                new_dist_to_collection = 0

        # --- 6. Termination Check ---
        terminated = False
        if self.crystals_collected >= self.NUM_CRYSTALS:
            terminated = True
            reward += 100.0
            self.score += 100
            self.last_event = "ALL CRYSTALS COLLECTED! +100"
            # sound: level_win.wav
        elif self.traps_triggered >= self.MAX_TRAPS_HIT:
            terminated = True
            reward -= 100.0
            self.score -= 100
            self.last_event = "TOO MANY TRAPS! -100"
            # sound: game_over.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        self.last_action_reward = reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

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
            "crystals_collected": self.crystals_collected,
            "traps_triggered": self.traps_triggered,
        }

    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                self._draw_iso_tile(self.screen, x, y, self.COLOR_BG, self.COLOR_GRID)

        # Draw collection zone
        cz_x, cz_y = self.collection_zone
        glow_alpha = 100 + 50 * math.sin(self.steps * 0.1)
        glow_color = (*self.COLOR_COLLECT_ZONE, glow_alpha)
        self._draw_iso_tile(self.screen, cz_x, cz_y, (30, 80, 80))
        # Draw a larger tile underneath for glow effect
        points = [
            self._grid_to_iso(cz_x - 0.2, cz_y - 0.2),
            self._grid_to_iso(cz_x + 1.2, cz_y - 0.2),
            self._grid_to_iso(cz_x + 1.2, cz_y + 1.2),
            self._grid_to_iso(cz_x - 0.2, cz_y + 1.2),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, glow_color)
        self._draw_iso_tile(self.screen, cz_x, cz_y, (0,0,0,0), self.COLOR_COLLECT_ZONE)


        # Draw traps
        for trap in self.traps:
            sx, sy = self._grid_to_iso(trap['pos'][0] + 0.5, trap['pos'][1] + 0.5)
            size = self.TILE_WIDTH_HALF * 0.6
            pygame.draw.line(self.screen, self.COLOR_TRAP, (sx - size, sy), (sx + size, sy), 3)
            pygame.draw.line(self.screen, self.COLOR_TRAP, (sx, sy - size/2), (sx, sy + size/2), 3)


        # Draw crystals
        for crystal in self.crystals:
            x, y = crystal['pos']
            color = crystal['color']
            
            # Main body
            self._draw_iso_tile(self.screen, x, y, color)
            
            # Highlight and shadow
            top_point = self._grid_to_iso(x, y)
            right_point = self._grid_to_iso(x + 1, y)
            bottom_point = self._grid_to_iso(x + 1, y + 1)
            left_point = self._grid_to_iso(x, y + 1)
            center_point = self._grid_to_iso(x + 0.5, y + 0.5)

            highlight_color = tuple(min(255, c + 50) for c in color)
            shadow_color = tuple(max(0, c - 50) for c in color)
            
            pygame.gfxdraw.filled_polygon(self.screen, [top_point, right_point, center_point], highlight_color)
            pygame.gfxdraw.filled_polygon(self.screen, [left_point, bottom_point, center_point], shadow_color)

            # Sparkle
            sparkle_size = 2 + 2 * math.sin(self.steps * 0.2 + crystal['id'])
            if sparkle_size > 2.5:
                pygame.draw.circle(self.screen, (255, 255, 255), center_point, int(sparkle_size))

        # Update and draw particles
        self._update_and_draw_particles()

    def _render_ui(self):
        def draw_text(text, pos, font, color=self.COLOR_TEXT, shadow_color=self.COLOR_TEXT_SHADOW):
            shadow_surf = font.render(text, True, shadow_color)
            text_surf = font.render(text, True, color)
            self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(text_surf, pos)
            
        # Top-left: Crystals
        draw_text(f"Crystals: {self.crystals_collected}/{self.NUM_CRYSTALS}", (20, 20), self.font_main)
        
        # Top-right: Traps
        trap_text = f"Traps Hit: {self.traps_triggered}/{self.MAX_TRAPS_HIT}"
        text_width = self.font_main.size(trap_text)[0]
        draw_text(trap_text, (self.SCREEN_WIDTH - text_width - 20, 20), self.font_main)

        # Bottom-center: Score
        score_text = f"Score: {self.score}"
        text_width = self.font_main.size(score_text)[0]
        draw_text(score_text, (self.SCREEN_WIDTH // 2 - text_width // 2, self.SCREEN_HEIGHT - 40), self.font_main)

        # Last event message
        if self.last_event:
            event_color = self.COLOR_TRAP if "TRAP" in self.last_event else self.COLOR_COLLECT_ZONE
            text_width = self.font_small.size(self.last_event)[0]
            draw_text(self.last_event, (self.SCREEN_WIDTH // 2 - text_width // 2, self.SCREEN_HEIGHT - 70), self.font_small, color=event_color)

        if self.game_over:
            result_text = "VICTORY!" if self.crystals_collected >= self.NUM_CRYSTALS else "GAME OVER"
            text_width = self.font_title.size(result_text)[0]
            draw_text(result_text, (self.SCREEN_WIDTH // 2 - text_width // 2, 200), self.font_title)

    def _create_particle(self, grid_pos, color, lifetime):
        sx, sy = self._grid_to_iso(grid_pos[0] + 0.5, grid_pos[1] + 0.5)
        return {
            "x": sx, "y": sy,
            "vx": self.np_random.uniform(-2, 2), "vy": self.np_random.uniform(-3, 1),
            "life": lifetime, "max_life": lifetime,
            "color": color
        }
    
    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p['life'] -= 1
            if p['life'] > 0:
                p['x'] += p['vx']
                p['y'] += p['vy']
                p['vy'] += 0.1 # Gravity
                
                size = (p['life'] / p['max_life']) * 5
                alpha = int((p['life'] / p['max_life']) * 255)
                color = (*p['color'], alpha)
                
                temp_surf = pygame.Surface((int(size*2), int(size*2)), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (int(size), int(size)), int(size))
                self.screen.blit(temp_surf, (p['x']-size, p['y']-size), special_flags=pygame.BLEND_RGBA_ADD)
                
                active_particles.append(p)
        self.particles = active_particles

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Running implementation validation...")
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
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    game_over_screen = False
    
    # Create a display for human interaction
    pygame.display.set_caption("Crystal Cavern")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    while running:
        action = np.array([0, 0, 0]) # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and not game_over_screen:
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
                    game_over_screen = False
                
                # Take a step if a move key was pressed
                if action[0] != 0:
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Step: {info['steps']}, Action: {action[0]}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
                    if terminated:
                        game_over_screen = True

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(30) # Limit FPS for human playability
        
    pygame.quit()