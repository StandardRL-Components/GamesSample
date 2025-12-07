
# Generated: 2025-08-27T18:51:30.046330
# Source Brief: brief_01970.md
# Brief Index: 1970

        
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
        "Controls: Use arrow keys to move the selector. Hold SPACE and press an arrow key to push a crystal."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric puzzle game. Push vibrant crystals onto pressure plates to activate them all before time runs out."
    )

    # Frames auto-advance for smooth graphics and time limits.
    auto_advance = True

    # --- Constants ---
    # Game parameters
    GRID_WIDTH = 22
    GRID_HEIGHT = 16
    TILE_WIDTH = 60
    TILE_HEIGHT = 30
    NUM_CRYSTALS = 10
    MAX_STEPS = 1800  # 60 seconds at 30 FPS

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_WALL = (40, 50, 70)
    COLOR_FLOOR = (60, 70, 90)
    COLOR_GRID = (70, 80, 100)
    COLOR_PLATE = (100, 100, 110)
    COLOR_PLATE_ACTIVE = (255, 255, 0)
    CRYSTAL_COLORS = [
        (255, 65, 54), (0, 116, 217), (46, 204, 64),
        (255, 133, 27), (177, 13, 201), (255, 220, 0),
        (57, 204, 204), (240, 18, 190)
    ]
    COLOR_BEAM = (255, 255, 0, 150)
    COLOR_SELECTOR = (255, 255, 255)
    COLOR_TEXT = (230, 230, 230)
    COLOR_TEXT_SHADOW = (10, 10, 10)

    # Rewards
    REWARD_GOAL = 100.0
    REWARD_FAIL = -100.0
    REWARD_NEW_CRYSTAL = 1.0
    REWARD_STEP = -0.01

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.screen_width = 640
        self.screen_height = 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Calculate offset to center the grid
        self.iso_offset_x = self.screen_width / 2
        self.iso_offset_y = (self.screen_height - self.GRID_HEIGHT * self.TILE_HEIGHT / 2) / 2 - 20

        # Initialize state variables
        self.grid = None
        self.crystals = None
        self.plates = None
        self.plate_crystal_map = None
        self.selector_pos = None
        self.steps = 0
        self.score = 0
        self.lit_crystals_last_step = 0
        self.game_over_message = ""
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over_message = ""
        
        self._generate_level()

        self.lit_crystals_last_step = self._count_lit_crystals()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = self.REWARD_STEP
        
        if not self.game_over_message:
            movement = action[0]
            space_held = action[1] == 1
            
            # --- Action Handling ---
            move_dir = self._get_move_dir(movement)
            
            if space_held and movement != 0:
                # Attempt to push a crystal
                crystal_idx = self._get_crystal_at(self.selector_pos)
                if crystal_idx is not None:
                    target_pos = [self.selector_pos[0] + move_dir[0], self.selector_pos[1] + move_dir[1]]
                    if self._is_valid_move(target_pos):
                        self.crystals[crystal_idx]['pos'] = target_pos
                        # sfx: crystal_push.wav
            else:
                # Move selector
                if movement != 0:
                    new_pos = [self.selector_pos[0] + move_dir[0], self.selector_pos[1] + move_dir[1]]
                    if self._is_on_grid(new_pos) and self.grid[new_pos[0]][new_pos[1]] == 0:
                        self.selector_pos = new_pos

        self.steps += 1
        
        # --- Update state and calculate rewards ---
        reward += self._calculate_reward()
        self.score += reward
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _calculate_reward(self):
        reward = 0
        current_lit_count = self._count_lit_crystals()
        newly_lit = current_lit_count - self.lit_crystals_last_step
        
        if newly_lit > 0:
            reward += self.REWARD_NEW_CRYSTAL * newly_lit
            # sfx: power_up.wav
        
        self.lit_crystals_last_step = current_lit_count
        return reward

    def _check_termination(self):
        if self.game_over_message: # Already terminated
            return True

        if self._count_lit_crystals() == self.NUM_CRYSTALS:
            self.score += self.REWARD_GOAL
            self.game_over_message = "SUCCESS!"
            # sfx: level_complete.wav
            return True
        
        if self.steps >= self.MAX_STEPS:
            self.score += self.REWARD_FAIL
            self.game_over_message = "TIME'S UP"
            # sfx: game_over.wav
            return True
            
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_cavern()
        self._render_objects()
        self._render_effects()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.MAX_STEPS - self.steps,
            "crystals_lit": self._count_lit_crystals(),
        }

    # --- Rendering Methods ---
    def _iso_to_screen(self, x, y):
        screen_x = (x - y) * (self.TILE_WIDTH / 2) + self.iso_offset_x
        screen_y = (x + y) * (self.TILE_HEIGHT / 2) + self.iso_offset_y
        return int(screen_x), int(screen_y)

    def _render_cavern(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                screen_pos = self._iso_to_screen(x, y)
                tile_points = [
                    self._iso_to_screen(x, y + 1),
                    self._iso_to_screen(x + 1, y + 1),
                    self._iso_to_screen(x + 1, y),
                    self._iso_to_screen(x, y),
                ]
                
                if self.grid[x][y] == 0:  # Floor
                    pygame.gfxdraw.filled_polygon(self.screen, tile_points, self.COLOR_FLOOR)
                    pygame.gfxdraw.aapolygon(self.screen, tile_points, self.COLOR_GRID)
                else: # Wall
                    wall_color = self.COLOR_WALL
                    top_points = [
                        self._iso_to_screen(x, y),
                        self._iso_to_screen(x + 1, y),
                        self._iso_to_screen(x + 1, y-1),
                        self._iso_to_screen(x, y-1),
                    ]
                    pygame.gfxdraw.filled_polygon(self.screen, tile_points, wall_color)
                    pygame.gfxdraw.filled_polygon(self.screen, top_points, tuple(min(255, c + 20) for c in wall_color))
                    pygame.gfxdraw.aapolygon(self.screen, tile_points, self.COLOR_GRID)
                    pygame.gfxdraw.aapolygon(self.screen, top_points, self.COLOR_GRID)

    def _render_objects(self):
        # Update lit status before drawing
        lit_crystals_indices = set()
        occupied_plates = set()
        for i, crystal in enumerate(self.crystals):
            for j, plate in enumerate(self.plates):
                if crystal['pos'] == plate['pos'] and j not in occupied_plates:
                    plate_activates_crystal_idx = self.plate_crystal_map[j]
                    lit_crystals_indices.add(plate_activates_crystal_idx)
                    occupied_plates.add(j)
                    break # A crystal can only occupy one plate

        for i in range(self.NUM_CRYSTALS):
            self.crystals[i]['lit'] = i in lit_crystals_indices

        # Draw in grid order for correct isometric layering
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                # Draw plates first
                for i, plate in enumerate(self.plates):
                    if plate['pos'] == [x, y]:
                        is_active = i in occupied_plates
                        self._draw_plate(plate['pos'], is_active)
                
                # Draw crystals on top
                for i, crystal in enumerate(self.crystals):
                    if crystal['pos'] == [x, y]:
                        self._draw_crystal(crystal)

    def _draw_plate(self, pos, is_active):
        screen_pos = self._iso_to_screen(pos[0] + 0.5, pos[1] + 0.5)
        color = self.COLOR_PLATE_ACTIVE if is_active else self.COLOR_PLATE
        radius = int(self.TILE_WIDTH / 4)
        pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], radius, color)
        pygame.gfxdraw.aacircle(self.screen, screen_pos[0], screen_pos[1], radius, tuple(min(255, c + 20) for c in color))

    def _draw_crystal(self, crystal):
        pos = crystal['pos']
        color = crystal['color']
        is_lit = crystal['lit']
        
        center_x, center_y = self._iso_to_screen(pos[0] + 0.5, pos[1] + 0.5)
        
        size = self.TILE_WIDTH / 4
        
        if is_lit:
            # Glow effect
            glow_color = color + (60,) # Add alpha
            glow_radius = int(size * 1.8)
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, glow_color, (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (center_x - glow_radius, center_y - glow_radius))

        # Crystal shape (isometric cube)
        top_points = [
            (center_x, center_y - size * 0.5),
            (center_x + size, center_y),
            (center_x, center_y + size * 0.5),
            (center_x - size, center_y)
        ]
        
        c_top = tuple(min(255, c+40) for c in color)
        c_left = color
        c_right = tuple(max(0, c-40) for c in color)
        
        # Left face
        pygame.gfxdraw.filled_polygon(self.screen, [top_points[3], top_points[2], (top_points[2][0], top_points[2][1]+size), (top_points[3][0], top_points[3][1]+size)], c_left)
        # Right face
        pygame.gfxdraw.filled_polygon(self.screen, [top_points[2], top_points[1], (top_points[1][0], top_points[1][1]+size), (top_points[2][0], top_points[2][1]+size)], c_right)
        # Top face
        pygame.gfxdraw.filled_polygon(self.screen, top_points, c_top)
        
        # Outline
        pygame.gfxdraw.aapolygon(self.screen, top_points, (0,0,0,100))

    def _render_effects(self):
        # Draw light beams
        for plate_idx, crystal_idx in self.plate_crystal_map.items():
            plate_pos = self.plates[plate_idx]['pos']
            crystal_on_plate = False
            for c in self.crystals:
                if c['pos'] == plate_pos:
                    crystal_on_plate = True
                    break
            
            if crystal_on_plate:
                start_pos = self._iso_to_screen(plate_pos[0] + 0.5, plate_pos[1] + 0.5)
                end_pos = self._iso_to_screen(self.crystals[crystal_idx]['pos'][0] + 0.5, self.crystals[crystal_idx]['pos'][1] + 0.5)
                pygame.draw.line(self.screen, self.COLOR_BEAM, start_pos, end_pos, 3)

        # Draw selector
        sel_x, sel_y = self.selector_pos
        tile_points = [
            self._iso_to_screen(sel_x, sel_y + 1),
            self._iso_to_screen(sel_x + 1, sel_y + 1),
            self._iso_to_screen(sel_x + 1, sel_y),
            self._iso_to_screen(sel_x, sel_y),
        ]
        pygame.draw.polygon(self.screen, self.COLOR_SELECTOR, tile_points, 3)

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos):
            shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(shadow, (pos[0] + 2, pos[1] + 2))
            surface = font.render(text, True, color)
            self.screen.blit(surface, pos)

        # Lit crystals count
        lit_count_text = f"ACTIVATED: {self._count_lit_crystals()} / {self.NUM_CRYSTALS}"
        draw_text(lit_count_text, self.font_small, self.COLOR_TEXT, (10, 10))
        
        # Timer
        time_left_sec = (self.MAX_STEPS - self.steps) / 30
        timer_text = f"TIME: {max(0, time_left_sec):.1f}"
        draw_text(timer_text, self.font_small, self.COLOR_TEXT, (self.screen_width - 120, 10))

        # Game Over Message
        if self.game_over_message:
            s = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            s.fill((0,0,0,128))
            self.screen.blit(s, (0,0))
            draw_text(self.game_over_message, self.font_large, self.COLOR_TEXT, 
                      (self.screen_width/2 - self.font_large.size(self.game_over_message)[0]/2, 
                       self.screen_height/2 - 50))
            score_text = f"Final Score: {self.score:.2f}"
            draw_text(score_text, self.font_small, self.COLOR_TEXT,
                      (self.screen_width/2 - self.font_small.size(score_text)[0]/2, 
                       self.screen_height/2))

    # --- Game Logic Helpers ---
    def _generate_level(self):
        self._generate_cavern()
        self._place_objects()
    
    def _generate_cavern(self):
        self.grid = np.ones((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        start_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.grid[start_pos[0]][start_pos[1]] = 0
        
        walkers = [{'pos': start_pos, 'life': 100}]
        floor_count = 1
        target_floor_count = (self.GRID_WIDTH * self.GRID_HEIGHT) * 0.6
        
        while floor_count < target_floor_count:
            if not walkers: break
            
            for walker in walkers.copy():
                move = random.choice([[0,1], [0,-1], [1,0], [-1,0]])
                new_pos = [walker['pos'][0] + move[0], walker['pos'][1] + move[1]]
                
                if self._is_on_grid(new_pos, margin=1):
                    if self.grid[new_pos[0]][new_pos[1]] == 1:
                        self.grid[new_pos[0]][new_pos[1]] = 0
                        floor_count += 1
                    walker['pos'] = new_pos
                    walker['life'] -= 1
                else:
                    walker['life'] = 0

                if random.random() < 0.1: # Chance to spawn new walker
                    walkers.append({'pos': walker['pos'][:], 'life': 50})

                if walker['life'] <= 0:
                    walkers.remove(walker)

    def _place_objects(self):
        floor_tiles = []
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[x][y] == 0:
                    floor_tiles.append([x, y])
        
        # Ensure enough space for all objects
        num_objects = self.NUM_CRYSTALS * 2 + 1 # Crystals, plates, selector
        if len(floor_tiles) < num_objects:
            # Fallback if cavern generation fails
            self._generate_level()
            return
            
        chosen_tiles = self.np_random.choice(len(floor_tiles), num_objects, replace=False)
        chosen_coords = [floor_tiles[i] for i in chosen_tiles]
        
        self.selector_pos = chosen_coords.pop()
        
        self.crystals = []
        crystal_coords = chosen_coords[:self.NUM_CRYSTALS]
        for i in range(self.NUM_CRYSTALS):
            self.crystals.append({
                'pos': crystal_coords[i],
                'color': random.choice(self.CRYSTAL_COLORS),
                'lit': False
            })

        self.plates = []
        plate_coords = chosen_coords[self.NUM_CRYSTALS:]
        for i in range(self.NUM_CRYSTALS):
            self.plates.append({'pos': plate_coords[i]})

        # Map plates to crystals they activate
        crystal_indices = list(range(self.NUM_CRYSTALS))
        self.np_random.shuffle(crystal_indices)
        self.plate_crystal_map = {i: crystal_indices[i] for i in range(self.NUM_CRYSTALS)}

    def _is_on_grid(self, pos, margin=0):
        return (margin <= pos[0] < self.GRID_WIDTH - margin and
                margin <= pos[1] < self.GRID_HEIGHT - margin)

    def _is_valid_move(self, pos):
        if not self._is_on_grid(pos): return False
        if self.grid[pos[0]][pos[1]] == 1: return False # Wall
        if self._get_crystal_at(pos) is not None: return False # Another crystal
        return True

    def _get_crystal_at(self, pos):
        for i, crystal in enumerate(self.crystals):
            if crystal['pos'] == pos:
                return i
        return None

    def _get_move_dir(self, movement):
        if movement == 1: return [0, -1]  # Up
        if movement == 2: return [0, 1]   # Down
        if movement == 3: return [-1, 0]  # Left
        if movement == 4: return [1, 0]   # Right
        return [0, 0]

    def _count_lit_crystals(self):
        # This function re-evaluates the lit state from scratch
        # to be the single source of truth.
        lit_count = 0
        occupied_plates = set()
        lit_crystals_indices = set()

        for crystal in self.crystals:
            for j, plate in enumerate(self.plates):
                if crystal['pos'] == plate['pos'] and j not in occupied_plates:
                    plate_activates_crystal_idx = self.plate_crystal_map[j]
                    lit_crystals_indices.add(plate_activates_crystal_idx)
                    occupied_plates.add(j)
                    break
        
        return len(lit_crystals_indices)
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.screen_height, self.screen_width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen_width, screen_height = 640, 400
    pygame_screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Crystal Caverns")
    
    running = True
    total_reward = 0.0
    
    # Game loop for human play
    while running:
        # --- Action mapping for human keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0.0

        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        pygame_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            print("Press 'R' to play again or close the window to exit.")
            
            # Wait for reset command
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        print("Resetting environment.")
                        obs, info = env.reset()
                        total_reward = 0.0
                        wait_for_reset = False
        
        # Control frame rate
        env.clock.tick(30)

    env.close()
    pygame.quit()