
# Generated: 2025-08-27T23:14:30.999233
# Source Brief: brief_03398.md
# Brief Index: 3398

        
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
        "Use Arrow Keys to push the selected crystal. Press Space/Shift to cycle through crystals."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric puzzle game. Push glowing crystals onto pressure plates to solve the puzzle against the clock."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    
    COLOR_BG = (20, 25, 40)
    COLOR_WALL = (60, 65, 80)
    COLOR_WALL_TOP = (80, 85, 100)
    COLOR_FLOOR = (40, 45, 60)
    
    COLOR_PLATE = (70, 70, 70)
    COLOR_PLATE_LIT = (120, 220, 255)
    
    COLOR_CRYSTAL = (50, 80, 180)
    COLOR_CRYSTAL_LIT = (100, 200, 255)
    COLOR_CRYSTAL_TOP = (80, 120, 220)
    COLOR_CRYSTAL_TOP_LIT = (150, 220, 255)
    
    COLOR_SELECT = (255, 255, 0)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_BG = (0, 0, 0, 128)

    TILE_W, TILE_H = 48, 24
    TILE_W_HALF, TILE_H_HALF = TILE_W // 2, TILE_H // 2
    BLOCK_HEIGHT = 20

    LEVEL_MAP = [
        "WWWWWWWWWWWW",
        "W.C.P.P.C.PW",
        "W..P.C.C.P.W",
        "W.C........W",
        "WP.C.WW.C.PW",
        "W.C..WW..C.W",
        "WP.C.WW.C.PW",
        "W.C........W",
        "W..P.C.C.P.W",
        "W.C.P.P.C.PW",
        "WWWWWWWWWWWW",
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 50)
        
        self.grid_offset_x = self.SCREEN_WIDTH // 2
        self.grid_offset_y = 80
        
        self.reset()
        
        # self.validate_implementation() # Uncomment for self-testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.termination_reason = ""
        self.timer = 60.0
        
        self._init_level()
        self.selected_crystal_index = 0
        self.last_action = (0, 0, 0)
        self.lit_crystals_count = 0
        
        self._update_crystal_states()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        last_movement, last_space, last_shift = self.last_action

        reward = -0.01  # Time penalty
        self.steps += 1
        self.timer -= 1.0 / self.FPS
        
        # --- Handle Actions ---
        space_pressed = space_held and not last_space
        shift_pressed = shift_held and not last_shift

        if space_pressed:
            self.selected_crystal_index = (self.selected_crystal_index + 1) % len(self.crystals)
        if shift_pressed:
            self.selected_crystal_index = (self.selected_crystal_index - 1 + len(self.crystals)) % len(self.crystals)

        # Handle push action
        if movement > 0:
            crystal = self.crystals[self.selected_crystal_index]
            if crystal['anim_progress'] == 0:
                # 1=up, 2=down, 3=left, 4=right -> isometric directions
                # Map to (dx, dy): 1: (0,-1), 2: (0,1), 3: (-1,0), 4: (1,0)
                dirs = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
                dx, dy = dirs[movement]
                
                if self._is_valid_push(self.selected_crystal_index, (dx, dy)):
                    # sfx: push_crystal.wav
                    crystal['target_pos'] = (crystal['pos'][0] + dx, crystal['pos'][1] + dy)
                    crystal['anim_progress'] = 1 / 10.0 # 10 frames to animate

        self.last_action = (movement, space_held, shift_held)

        # --- Update Game State ---
        for c in self.crystals:
            if c['anim_progress'] > 0:
                c['anim_progress'] += 1 / 10.0
                if c['anim_progress'] >= 1.0:
                    c['anim_progress'] = 0
                    c['pos'] = c['target_pos']
                    
                    if not self._check_solvability(c):
                        self.game_over = True
                        self.termination_reason = "Crystal Stuck!"
                        reward -= 100
                    
                    self._update_crystal_states()
        
        new_lit_count = sum(1 for c in self.crystals if c['lit'])
        if new_lit_count > self.lit_crystals_count:
            # sfx: crystal_lit.wav
            reward += (new_lit_count - self.lit_crystals_count) * 1.0
            self.lit_crystals_count = new_lit_count

        self.score += reward
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            if self.lit_crystals_count == len(self.crystals):
                # sfx: victory.wav
                self.termination_reason = "All Crystals Lit!"
                reward += 100
            elif self.timer <= 0:
                # sfx: game_over.wav
                self.termination_reason = "Time's Up!"
                reward -= 100
            self.game_over = True
            self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
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
            "timer": self.timer,
            "lit_crystals": self.lit_crystals_count,
            "total_crystals": len(self.crystals),
        }

    # --- Rendering Methods ---
    def _render_game(self):
        # Sort all objects for correct isometric drawing order
        drawable_objects = []
        for y in range(len(self.LEVEL_MAP)):
            for x in range(len(self.LEVEL_MAP[0])):
                drawable_objects.append({'type': 'floor', 'pos': (x, y)})
                if (x, y) in self.plates:
                    drawable_objects.append({'type': 'plate', 'pos': (x, y)})
                if (x, y) in self.walls:
                    drawable_objects.append({'type': 'wall', 'pos': (x, y)})

        for i, c in enumerate(self.crystals):
            drawable_objects.append({'type': 'crystal', 'index': i, 'pos': c['pos']})
        
        drawable_objects.sort(key=lambda obj: (obj['pos'][0] + obj['pos'][1], obj['pos'][1], obj['pos'][0]))

        for obj in drawable_objects:
            if obj['type'] == 'floor':
                self._draw_iso_rect(self.screen, self.COLOR_FLOOR, obj['pos'], self.TILE_W, self.TILE_H)
            elif obj['type'] == 'plate':
                is_lit = any(c['pos'] == obj['pos'] for c in self.crystals)
                color = self.COLOR_PLATE_LIT if is_lit else self.COLOR_PLATE
                center_x, center_y = self._grid_to_iso(obj['pos'][0], obj['pos'][1])
                pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, self.TILE_H_HALF - 2, color)
                pygame.gfxdraw.aacircle(self.screen, center_x, center_y, self.TILE_H_HALF - 2, color)
            elif obj['type'] == 'wall':
                self._draw_iso_block(self.screen, self.COLOR_WALL, self.COLOR_WALL_TOP, obj['pos'])
            elif obj['type'] == 'crystal':
                self._draw_crystal(self.crystals[obj['index']], obj['index'] == self.selected_crystal_index)

    def _render_ui(self):
        # UI Background
        ui_surface = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        ui_surface.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surface, (0, 0))

        # Lit Crystals
        lit_text = f"CRYSTALS: {self.lit_crystals_count} / {len(self.crystals)}"
        text_surf = self.font_ui.render(lit_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (10, 10))

        # Timer
        timer_text = f"TIME: {max(0, self.timer):.1f}"
        text_surf = self.font_ui.render(timer_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (self.SCREEN_WIDTH - text_surf.get_width() - 10, 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0,0,0,180))
            self.screen.blit(overlay, (0,0))
            
            text_surf = self.font_game_over.render(self.termination_reason, True, self.COLOR_UI_TEXT)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    # --- Helper Methods ---
    def _init_level(self):
        self.walls = set()
        self.crystals = []
        self.plates = set()
        for y, row in enumerate(self.LEVEL_MAP):
            for x, char in enumerate(row):
                pos = (x, y)
                if char == 'W':
                    self.walls.add(pos)
                elif char == 'C':
                    self.crystals.append({'pos': pos, 'target_pos': pos, 'lit': False, 'anim_progress': 0.0})
                elif char == 'P':
                    self.plates.add(pos)
    
    def _grid_to_iso(self, x, y):
        iso_x = self.grid_offset_x + (x - y) * self.TILE_W_HALF
        iso_y = self.grid_offset_y + (x + y) * self.TILE_H_HALF
        return int(iso_x), int(iso_y)

    def _draw_iso_rect(self, surface, color, pos, w, h):
        x, y = self._grid_to_iso(pos[0], pos[1])
        points = [
            (x, y - h // 2),
            (x + w // 2, y),
            (x, y + h // 2),
            (x - w // 2, y)
        ]
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def _draw_iso_block(self, surface, side_color, top_color, pos):
        x, y = self._grid_to_iso(pos[0], pos[1])
        y -= self.BLOCK_HEIGHT // 2
        
        top_points = [
            (x, y - self.TILE_H_HALF),
            (x + self.TILE_W_HALF, y),
            (x, y + self.TILE_H_HALF),
            (x - self.TILE_W_HALF, y)
        ]
        
        right_side = [
            (x + self.TILE_W_HALF, y),
            (x + self.TILE_W_HALF, y + self.BLOCK_HEIGHT),
            (x, y + self.TILE_H_HALF + self.BLOCK_HEIGHT),
            (x, y + self.TILE_H_HALF)
        ]
        
        left_side = [
            (x - self.TILE_W_HALF, y),
            (x - self.TILE_W_HALF, y + self.BLOCK_HEIGHT),
            (x, y + self.TILE_H_HALF + self.BLOCK_HEIGHT),
            (x, y + self.TILE_H_HALF)
        ]

        pygame.gfxdraw.filled_polygon(surface, right_side, side_color)
        pygame.gfxdraw.filled_polygon(surface, left_side, side_color)
        pygame.gfxdraw.filled_polygon(surface, top_points, top_color)
        pygame.gfxdraw.aapolygon(surface, top_points, top_color)

    def _draw_crystal(self, crystal, is_selected):
        anim_prog = crystal['anim_progress']
        start_pos = crystal['pos']
        target_pos = crystal['target_pos']

        # Interpolate grid position for smooth animation
        interp_x = start_pos[0] + (target_pos[0] - start_pos[0]) * anim_prog
        interp_y = start_pos[1] + (target_pos[1] - start_pos[1]) * anim_prog
        
        # Determine colors and glow
        top_color = self.COLOR_CRYSTAL_TOP_LIT if crystal['lit'] else self.COLOR_CRYSTAL_TOP
        side_color = self.COLOR_CRYSTAL_LIT if crystal['lit'] else self.COLOR_CRYSTAL
        
        # Pulsing glow for lit crystals
        if crystal['lit']:
            pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
            glow_color = tuple(min(255, int(c + pulse * 30)) for c in side_color)
            glow_top_color = tuple(min(255, int(c + pulse * 30)) for c in top_color)
            side_color, top_color = glow_color, glow_top_color

        # Draw block
        x, y = self._grid_to_iso(interp_x, interp_y)
        y -= self.BLOCK_HEIGHT // 2
        
        top_points = [
            (x, y - self.TILE_H_HALF), (x + self.TILE_W_HALF, y),
            (x, y + self.TILE_H_HALF), (x - self.TILE_W_HALF, y)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, top_points, top_color)
        
        right_side = [(x + self.TILE_W_HALF, y), (x + self.TILE_W_HALF, y + self.BLOCK_HEIGHT), (x, y + self.TILE_H_HALF + self.BLOCK_HEIGHT), (x, y + self.TILE_H_HALF)]
        left_side = [(x - self.TILE_W_HALF, y), (x - self.TILE_W_HALF, y + self.BLOCK_HEIGHT), (x, y + self.TILE_H_HALF + self.BLOCK_HEIGHT), (x, y + self.TILE_H_HALF)]
        
        pygame.gfxdraw.filled_polygon(self.screen, right_side, side_color)
        pygame.gfxdraw.filled_polygon(self.screen, left_side, side_color)
        pygame.gfxdraw.aapolygon(self.screen, top_points, top_color)

        # Draw selection highlight
        if is_selected:
            sel_points = [(p[0], p[1]-1) for p in top_points]
            pygame.draw.aalines(self.screen, self.COLOR_SELECT, True, sel_points, 2)
            sel_points2 = [(p[0], p[1]-2) for p in top_points]
            pygame.draw.aalines(self.screen, self.COLOR_SELECT, True, sel_points2, 1)

    def _is_valid_push(self, crystal_index, direction):
        crystal = self.crystals[crystal_index]
        target_pos = (crystal['pos'][0] + direction[0], crystal['pos'][1] + direction[1])

        if target_pos in self.walls:
            return False
        
        for i, other_crystal in enumerate(self.crystals):
            if i != crystal_index and other_crystal['pos'] == target_pos:
                return False
        
        return True

    def _update_crystal_states(self):
        for c in self.crystals:
            c['lit'] = c['pos'] in self.plates

    def _check_solvability(self, crystal):
        if crystal['pos'] in self.plates:
            return True # A crystal on a plate is never a losing move

        x, y = crystal['pos']
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        blocked_sides = 0
        
        crystal_positions = {c['pos'] for c in self.crystals}

        for nx, ny in neighbors:
            if (nx, ny) in self.walls or (nx, ny) in crystal_positions:
                blocked_sides += 1
        
        return blocked_sides < 4

    def _check_termination(self):
        if self.game_over:
            return True
        if self.timer <= 0:
            return True
        if self.lit_crystals_count == len(self.crystals):
            return True
        if self.steps >= 1800: # 60 seconds * 30 fps
            return True
        return False

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
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Crystal Caverns")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")
                if event.key == pygame.K_ESCAPE:
                    running = False

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Episode Finished! Final Score: {info['score']:.2f}, Reason: {env.termination_reason}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(env.FPS)

    env.close()
    pygame.quit()