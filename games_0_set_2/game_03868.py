
# Generated: 2025-08-28T00:41:28.253343
# Source Brief: brief_03868.md
# Brief Index: 3868

        
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
        "Controls: Use arrow keys to move the cursor. Press space to place a reflecting crystal."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Illuminate all gems in a dark cavern by strategically placing light-reflecting crystals."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 20
    NUM_GEMS = 20
    INITIAL_CRYSTALS = 25 # Increased from brief's 15 to make it more forgiving
    MAX_STEPS = 1000
    GEM_RADIUS = 8
    CRYSTAL_SIZE = 8
    
    # --- Colors ---
    COLOR_BG = (10, 20, 40)
    COLOR_CURSOR = (0, 255, 255, 100)
    COLOR_CRYSTAL = (0, 220, 255)
    COLOR_CRYSTAL_GLOW = (0, 220, 255, 50)
    COLOR_GEM_UNLIT = (200, 30, 30)
    COLOR_GEM_UNLIT_GLOW = (200, 30, 30, 40)
    COLOR_GEM_LIT = (255, 255, 0)
    COLOR_GEM_LIT_GLOW = (255, 255, 0, 100)
    COLOR_LIGHT_BEAM = (255, 255, 240)
    COLOR_LIGHT_GLOW = (255, 255, 240, 60)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_WIN_TEXT = (100, 255, 100)
    COLOR_LOSE_TEXT = (255, 100, 100)

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
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.cursor_surface = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
        self.cursor_surface.fill(self.COLOR_CURSOR)

        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_lit_gem_count = 0
        self.crystals_remaining = 0
        self.cursor_pos = [0, 0]
        self.light_source = {}
        self.crystals = []
        self.gems = []
        self.light_paths = []

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_lit_gem_count = 0
        self.crystals_remaining = self.INITIAL_CRYSTALS
        
        self.cursor_pos = [
            (self.SCREEN_WIDTH // self.GRID_SIZE // 2) * self.GRID_SIZE,
            (self.SCREEN_HEIGHT // self.GRID_SIZE // 2) * self.GRID_SIZE
        ]
        
        source_y = (self.np_random.integers(2, self.SCREEN_HEIGHT // self.GRID_SIZE - 2)) * self.GRID_SIZE
        self.light_source = {'pos': [0, source_y + self.GRID_SIZE // 2], 'dir': [1, 0]}

        self.crystals = []
        self._generate_gems()
        
        self._calculate_light_paths()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        space_held = action[1] == 1
        
        reward = -0.01 # Small penalty for taking a step
        self.steps += 1

        self._move_cursor(movement)

        placed_crystal_this_step = False
        if space_held:
            if self._place_crystal():
                placed_crystal_this_step = True
                # sfx: crystal_place.wav
            else:
                reward -= 0.1 # Penalty for failed placement

        if placed_crystal_this_step:
            current_lit_count = sum(1 for gem in self.gems if gem['lit'])
            newly_lit = current_lit_count - self.last_lit_gem_count
            
            if newly_lit > 0:
                reward += newly_lit * 5 # +5 for each new gem
                # sfx: gem_lit.wav
            
            self.last_lit_gem_count = current_lit_count

        terminated = self._check_termination()
        
        if terminated:
            if self.last_lit_gem_count == self.NUM_GEMS:
                reward += 100 # Win bonus
                # sfx: win.wav
            elif self.crystals_remaining <= 0:
                reward -= 100 # Lose penalty
                # sfx: lose.wav

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_gems(self):
        self.gems = []
        occupied_cells = set()
        
        # Reserve light source start path
        initial_y_cell = self.light_source['pos'][1] // self.GRID_SIZE
        for i in range(self.SCREEN_WIDTH // self.GRID_SIZE):
            occupied_cells.add((i, initial_y_cell))
            
        while len(self.gems) < self.NUM_GEMS:
            gx = self.np_random.integers(2, self.SCREEN_WIDTH // self.GRID_SIZE - 2)
            gy = self.np_random.integers(1, self.SCREEN_HEIGHT // self.GRID_SIZE - 1)
            
            if (gx, gy) not in occupied_cells:
                occupied_cells.add((gx, gy))
                pos = [gx * self.GRID_SIZE + self.GRID_SIZE // 2, gy * self.GRID_SIZE + self.GRID_SIZE // 2]
                self.gems.append({'pos': pos, 'lit': False, 'radius': self.GEM_RADIUS})

    def _move_cursor(self, movement):
        if movement == 1: # Up
            self.cursor_pos[1] -= self.GRID_SIZE
        elif movement == 2: # Down
            self.cursor_pos[1] += self.GRID_SIZE
        elif movement == 3: # Left
            self.cursor_pos[0] -= self.GRID_SIZE
        elif movement == 4: # Right
            self.cursor_pos[0] += self.GRID_SIZE
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH - self.GRID_SIZE)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT - self.GRID_SIZE)

    def _place_crystal(self):
        if self.crystals_remaining <= 0:
            return False

        center_pos = [self.cursor_pos[0] + self.GRID_SIZE // 2, self.cursor_pos[1] + self.GRID_SIZE // 2]
        
        for obj_list in [self.gems, self.crystals]:
            for obj in obj_list:
                dist = math.hypot(obj['pos'][0] - center_pos[0], obj['pos'][1] - center_pos[1])
                if dist < self.GRID_SIZE: # Prevent placing on top of other objects
                    return False
        
        self.crystals.append({'pos': center_pos, 'radius': self.CRYSTAL_SIZE})
        self.crystals_remaining -= 1
        self._calculate_light_paths()
        return True

    def _calculate_light_paths(self):
        self.light_paths = []
        for gem in self.gems:
            gem['lit'] = False

        q = [(self.light_source['pos'], self.light_source['dir'])]
        processed_rays = set()
        
        max_bounces = 50 # Safety break
        iterations = 0

        while q and iterations < max_bounces:
            start_pos, direction = q.pop(0)
            iterations += 1

            ray_tuple = (tuple(start_pos), tuple(direction))
            if ray_tuple in processed_rays:
                continue
            processed_rays.add(ray_tuple)

            obstacles = self.gems + self.crystals
            
            min_dist = float('inf')
            closest_obj = None

            for obj in obstacles:
                # Simplified Axis-Aligned Bounding Box check for ray intersection
                dist = float('inf')
                if direction[0] != 0: # Horizontal ray
                    if abs(start_pos[1] - obj['pos'][1]) <= obj['radius']:
                        if (obj['pos'][0] - start_pos[0]) * direction[0] > 0:
                            dist = abs(obj['pos'][0] - start_pos[0])
                else: # Vertical ray
                    if abs(start_pos[0] - obj['pos'][0]) <= obj['radius']:
                        if (obj['pos'][1] - start_pos[1]) * direction[1] > 0:
                            dist = abs(obj['pos'][1] - start_pos[1])
                
                if dist < min_dist:
                    min_dist = dist
                    closest_obj = obj
            
            end_pos = [start_pos[0] + direction[0] * min_dist, start_pos[1] + direction[1] * min_dist]

            if closest_obj:
                self.light_paths.append((start_pos, end_pos))
                if 'lit' in closest_obj: # It's a gem
                    closest_obj['lit'] = True
                else: # It's a crystal
                    # sfx: reflect.wav
                    if direction[0] != 0: # Horizontal ray reflects vertically
                        q.append((end_pos, [0, 1]))
                        q.append((end_pos, [0, -1]))
                    else: # Vertical ray reflects horizontally
                        q.append((end_pos, [1, 0]))
                        q.append((end_pos, [-1, 0]))
            else: # Ray goes to edge of screen
                if direction[0] > 0: end_pos = [self.SCREEN_WIDTH, start_pos[1]]
                elif direction[0] < 0: end_pos = [0, start_pos[1]]
                elif direction[1] > 0: end_pos = [start_pos[0], self.SCREEN_HEIGHT]
                else: end_pos = [start_pos[0], 0]
                self.light_paths.append((start_pos, end_pos))

    def _check_termination(self):
        all_gems_lit = self.last_lit_gem_count == self.NUM_GEMS
        out_of_crystals = self.crystals_remaining <= 0
        
        if all_gems_lit or (out_of_crystals and not all_gems_lit) or self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render light source glow
        pygame.gfxdraw.filled_circle(self.screen, int(self.light_source['pos'][0]), int(self.light_source['pos'][1]), 12, self.COLOR_LIGHT_GLOW)

        # Render light paths
        for start, end in self.light_paths:
            pygame.draw.line(self.screen, self.COLOR_LIGHT_GLOW, start, end, 5)
            pygame.draw.aaline(self.screen, self.COLOR_LIGHT_BEAM, start, end, 1)

        # Render gems
        for gem in self.gems:
            color = self.COLOR_GEM_LIT if gem['lit'] else self.COLOR_GEM_UNLIT
            glow_color = self.COLOR_GEM_LIT_GLOW if gem['lit'] else self.COLOR_GEM_UNLIT_GLOW
            pos = (int(gem['pos'][0]), int(gem['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.GEM_RADIUS + 4, glow_color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.GEM_RADIUS, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.GEM_RADIUS, color)

        # Render crystals
        for crystal in self.crystals:
            pos = (int(crystal['pos'][0]), int(crystal['pos'][1]))
            size = self.CRYSTAL_SIZE
            points = [(pos[0], pos[1] - size), (pos[0] + size, pos[1]), (pos[0], pos[1] + size), (pos[0] - size, pos[1])]
            glow_points = [(pos[0], pos[1] - size-3), (pos[0] + size+3, pos[1]), (pos[0], pos[1] + size+3), (pos[0] - size-3, pos[1])]
            pygame.gfxdraw.filled_polygon(self.screen, glow_points, self.COLOR_CRYSTAL_GLOW)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_CRYSTAL)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_CRYSTAL)

        # Render cursor
        if not self.game_over:
            self.screen.blit(self.cursor_surface, self.cursor_pos)
    
    def _render_ui(self):
        # Render crystals remaining
        crystal_text = self.font_small.render(f"CRYSTALS: {self.crystals_remaining}", True, self.COLOR_UI_TEXT)
        self.screen.blit(crystal_text, (10, 10))

        # Render score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Render lit gems count
        lit_text = self.font_small.render(f"GEMS LIT: {self.last_lit_gem_count}/{self.NUM_GEMS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lit_text, (10, 30))

        if self.game_over:
            if self.last_lit_gem_count == self.NUM_GEMS:
                msg = "SUCCESS"
                color = self.COLOR_WIN_TEXT
            else:
                msg = "FAILURE"
                color = self.COLOR_LOSE_TEXT
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "crystals_remaining": self.crystals_remaining,
            "gems_lit": self.last_lit_gem_count,
        }

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

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Crystal Cavern")
    
    terminated = False
    clock = pygame.time.Clock()
    
    while not terminated:
        movement = 0 # No-op
        space_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
            
        action = [movement, space_held, 0] # shift is unused
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # In interactive mode, we need to manually update the display
        # and manage the game loop based on actions.
        # Since auto_advance is False, the game only steps when we call env.step().
        # We need a way to trigger a step. Let's step on any key press.
        
        # The default loop is too fast for a turn-based game.
        # Let's change the logic to only step on a key event.
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Wait for an action to be taken to advance the frame
        action_taken = False
        while not action_taken and not terminated:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                    action_taken = True
                if event.type == pygame.KEYDOWN:
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_UP]: movement = 1
                    elif keys[pygame.K_DOWN]: movement = 2
                    elif keys[pygame.K_LEFT]: movement = 3
                    elif keys[pygame.K_RIGHT]: movement = 4
                    else: movement = 0
                    
                    space_held = 1 if keys[pygame.K_SPACE] else 0
                    
                    action = [movement, space_held, 0]
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                    screen.blit(surf, (0, 0))
                    pygame.display.flip()

                    print(f"Step: {info['steps']}, Score: {info['score']:.2f}, Reward: {reward:.2f}, Gems: {info['gems_lit']}/{env.NUM_GEMS}")
                    
                    if terminated:
                        print("Game Over!")

            clock.tick(30) # Keep the window responsive

    # Keep the final screen visible for a few seconds
    if terminated:
        pygame.time.wait(3000)

    env.close()