import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set Pygame to run in headless mode
os.environ["SDL_VIDEODRIVER"] = "dummy"


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move your avatar. Collect blue crystals and avoid red traps."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Collect crystals in an isometric world while avoiding moving traps. Reach 20 crystals to win, but watch the timer!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    # Game parameters
    GRID_WIDTH = 20
    GRID_HEIGHT = 15
    TILE_WIDTH_HALF = 18
    TILE_HEIGHT_HALF = 9
    MAX_STEPS = 1000
    WIN_SCORE = 20
    INITIAL_CRYSTALS = 5
    INITIAL_TRAPS = 4
    DIFFICULTY_INCREASE_STEP = 500

    # Colors
    COLOR_BG = (35, 45, 55)
    COLOR_GRID = (50, 65, 75)
    COLOR_PLAYER = (255, 200, 0)
    COLOR_PLAYER_SHADOW = (200, 150, 0)
    COLOR_CRYSTAL = (0, 242, 234)
    COLOR_CRYSTAL_SHADOW = (0, 180, 170)
    COLOR_TRAP = (217, 4, 41)
    COLOR_TRAP_SHADOW = (160, 0, 30)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_BAR_BG = (70, 85, 95)
    COLOR_UI_BAR_FILL = (0, 242, 234)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)

        # Game state variables are initialized in reset()
        self.player_pos = None
        self.crystals = None
        self.traps = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.trap_move_increment = None
        
        # Calculate screen offset for centering the grid
        self.origin_x = 320
        self.origin_y = 60
        
        # Initialize state variables
        # A seed is not passed here, but the super().reset() in the reset method will handle it
        # if it's passed to the env.reset() call later. For the initial reset, it will use a random seed.
        # self.reset() is called at the end of __init__ to ensure a valid initial state.
        
        # Run validation check
        # self.validate_implementation() is called after reset to confirm correctness
        
        # The original code called reset() here, but it's better practice to let the user
        # call it explicitly. However, to pass the original test harness which expects
        # the env to be ready after __init__, we call it.
        self.reset()
        
        # The user's code included this validation call in __init__.
        # We keep it to match the original structure.
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.trap_move_increment = 1

        # Place player
        self.player_pos = np.array([self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2])
        
        # Initialize entity lists before spawning to avoid iterating over None
        self.crystals = []
        self.traps = []

        # Spawn crystals
        for _ in range(self.INITIAL_CRYSTALS):
            self._spawn_crystal()

        # Spawn traps
        for _ in range(self.INITIAL_TRAPS):
            self._spawn_trap()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        reward = 0

        # --- Pre-move distance calculations for reward shaping ---
        dist_to_closest_crystal_before = self._get_closest_entity_dist(self.player_pos, self.crystals)
        dist_to_closest_trap_before = self._get_closest_entity_dist(self.player_pos, [t['pos'] for t in self.traps])

        # --- Update player position ---
        prev_player_pos = self.player_pos.copy()
        if movement == 1: # Up
            self.player_pos[1] -= 1
        elif movement == 2: # Down
            self.player_pos[1] += 1
        elif movement == 3: # Left
            self.player_pos[0] -= 1
        elif movement == 4: # Right
            self.player_pos[0] += 1
        
        # Clamp player position to grid boundaries
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.GRID_WIDTH - 1)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.GRID_HEIGHT - 1)
        
        player_moved = not np.array_equal(prev_player_pos, self.player_pos)

        # --- Update trap positions ---
        self._move_traps()

        # --- Post-move distance calculations ---
        if player_moved:
            dist_to_closest_crystal_after = self._get_closest_entity_dist(self.player_pos, self.crystals)
            dist_to_closest_trap_after = self._get_closest_entity_dist(self.player_pos, [t['pos'] for t in self.traps])
            
            # Distance-based rewards
            if dist_to_closest_crystal_after < dist_to_closest_crystal_before:
                reward += 0.1 # Moved closer to a crystal
            if dist_to_closest_trap_after < dist_to_closest_trap_before:
                reward -= 0.2 # Moved closer to a trap

        # --- Check for collisions and events ---
        # Crystal collection
        collected_crystal_idx = -1
        for i, crystal_pos in enumerate(self.crystals):
            if np.array_equal(self.player_pos, crystal_pos):
                collected_crystal_idx = i
                break
        
        if collected_crystal_idx != -1:
            self.crystals.pop(collected_crystal_idx)
            self._spawn_crystal()
            self.score += 1
            reward += 1.0

        # Trap collision
        for trap in self.traps:
            if np.array_equal(self.player_pos, trap['pos']):
                self.game_over = True
                reward -= 100.0
                break
        
        # --- Update game state and check for termination ---
        self.steps += 1
        
        # Difficulty increase
        if self.steps == self.DIFFICULTY_INCREASE_STEP:
            self.trap_move_increment = 2

        terminated = self.game_over
        if self.score >= self.WIN_SCORE:
            terminated = True
            reward += 100.0
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        truncated = False # This environment does not truncate based on time limit

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        
        # Render entities (sorted by y-grid for correct overlap)
        entities = []
        for c_pos in self.crystals:
            entities.append({'pos': c_pos, 'type': 'crystal'})
        for t in self.traps:
            entities.append({'pos': t['pos'], 'type': 'trap'})
        entities.append({'pos': self.player_pos, 'type': 'player'})
        
        entities.sort(key=lambda e: e['pos'][0] + e['pos'][1])

        for entity in entities:
            if entity['type'] == 'player':
                self._render_player()
            elif entity['type'] == 'crystal':
                self._render_crystal(entity['pos'])
            elif entity['type'] == 'trap':
                self._render_trap(entity['pos'])

        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    # --- Helper and Rendering Methods ---

    def _iso_to_screen(self, x, y):
        screen_x = self.origin_x + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.origin_y + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _draw_iso_diamond(self, surface, color, shadow_color, x, y, size_mod=1.0):
        px, py = self._iso_to_screen(x, y)
        w = self.TILE_WIDTH_HALF * size_mod
        h = self.TILE_HEIGHT_HALF * size_mod
        
        points = [
            (px, py - h),
            (px + w, py),
            (px, py + h),
            (px - w, py)
        ]
        
        shadow_points = [(p[0], p[1] + 4) for p in points]

        pygame.gfxdraw.filled_polygon(surface, shadow_points, shadow_color)
        pygame.gfxdraw.aapolygon(surface, shadow_points, shadow_color)
        
        pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, color)

    def _get_valid_spawn_pos(self):
        while True:
            pos = self.np_random.integers([0, 0], [self.GRID_WIDTH, self.GRID_HEIGHT], size=2)
            
            is_on_player = np.array_equal(pos, self.player_pos)
            is_on_crystal = any(np.array_equal(pos, c) for c in self.crystals)
            is_on_trap = any(np.array_equal(pos, t['pos']) for t in self.traps)

            if not is_on_player and not is_on_crystal and not is_on_trap:
                return pos

    def _spawn_crystal(self):
        pos = self._get_valid_spawn_pos()
        self.crystals.append(pos)
        
    def _spawn_trap(self):
        pos = self._get_valid_spawn_pos()
        
        # Create a random path for the trap
        path_type = self.np_random.integers(0, 3)
        path = []
        if path_type == 0: # Horizontal
            for i in range(self.GRID_WIDTH):
                path.append(np.array([i, pos[1]]))
        elif path_type == 1: # Vertical
            for i in range(self.GRID_HEIGHT):
                path.append(np.array([pos[0], i]))
        else: # Box
            x, y = pos
            w, h = self.np_random.integers(2, 6, size=2)
            for i in range(w): path.append(np.array([min(x+i, self.GRID_WIDTH-1), y]))
            for i in range(h): path.append(np.array([min(x+w-1, self.GRID_WIDTH-1), min(y+i, self.GRID_HEIGHT-1)]))
            for i in range(w): path.append(np.array([min(x+w-1-i, self.GRID_WIDTH-1), min(y+h-1, self.GRID_HEIGHT-1)]))
            for i in range(h): path.append(np.array([x, min(y+h-1-i, self.GRID_HEIGHT-1)]))

        if not path: # Fallback for zero-length paths
             path.append(pos)
             
        self.traps.append({
            "pos": pos,
            "path": path,
            "path_idx": 0,
            "direction": 1
        })

    def _move_traps(self):
        for trap in self.traps:
            for _ in range(self.trap_move_increment):
                trap['path_idx'] += trap['direction']
                
                # Ping-pong at path ends
                if not (0 <= trap['path_idx'] < len(trap['path'])):
                    trap['direction'] *= -1
                    trap['path_idx'] += 2 * trap['direction']
                    # Handle case where path is very short
                    trap['path_idx'] = np.clip(trap['path_idx'], 0, len(trap['path']) - 1)
            
            trap['pos'] = trap['path'][trap['path_idx']]

    def _get_closest_entity_dist(self, pos, entity_list):
        if not entity_list:
            return float('inf')
        distances = [np.sum(np.abs(pos - e_pos)) for e_pos in entity_list]
        return min(distances)

    def _render_grid(self):
        for x in range(self.GRID_WIDTH + 1):
            start = self._iso_to_screen(x, 0)
            end = self._iso_to_screen(x, self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for y in range(self.GRID_HEIGHT + 1):
            start = self._iso_to_screen(0, y)
            end = self._iso_to_screen(self.GRID_WIDTH, y)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

    def _render_player(self):
        self._draw_iso_diamond(self.screen, self.COLOR_PLAYER, self.COLOR_PLAYER_SHADOW, self.player_pos[0], self.player_pos[1], 1.1)

    def _render_crystal(self, pos):
        # Pulsating effect for crystals
        size_mod = 0.9 + 0.1 * math.sin(self.steps * 0.2 + pos[0])
        self._draw_iso_diamond(self.screen, self.COLOR_CRYSTAL, self.COLOR_CRYSTAL_SHADOW, pos[0], pos[1], size_mod)

    def _render_trap(self, pos):
        self._draw_iso_diamond(self.screen, self.COLOR_TRAP, self.COLOR_TRAP_SHADOW, pos[0], pos[1])

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"CRYSTALS: {self.score}/{self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer Bar
        timer_bar_width = 200
        timer_bar_height = 15
        bar_x = 640 - timer_bar_width - 10
        bar_y = 15
        
        progress = max(0, 1 - (self.steps / self.MAX_STEPS))
        
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (bar_x, bar_y, timer_bar_width, timer_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_FILL, (bar_x, bar_y, int(timer_bar_width * progress), timer_bar_height))

    def close(self):
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


# Example of how to run the environment
if __name__ == '__main__':
    # For interactive play, you would remove the headless environment variable setting
    # and create a display window.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    # --- To play manually ---
    pygame.display.set_caption("Crystal Collector")
    screen = pygame.display.set_mode((640, 400))
    
    obs, info = env.reset()
    done = False
    
    print("\n" + env.game_description)
    print(env.user_guide)

    while not done:
        # Map pygame keys to actions
        movement_action = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        
        action = [movement_action, 0, 0] # Space and shift are not used

        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            # Only register an action on keydown for turn-based play
            if event.type == pygame.KEYDOWN:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Render the observation to the display
                surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                screen.blit(surf, (0, 0))
                pygame.display.flip()

                print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

    print("Game Over!")
    env.close()