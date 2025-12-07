
# Generated: 2025-08-27T14:36:49.762000
# Source Brief: brief_00738.md
# Brief Index: 738

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
    Crystal Cavern is an isometric puzzle game where the player must push crystals
    onto pressure plates to illuminate them. The goal is to light up all crystals
    within the time limit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Arrow keys to push the selected crystal. Space/Shift to cycle selection."
    )

    # User-facing description of the game
    game_description = (
        "A timed isometric puzzle. Push crystals onto pressure plates to light them up. Illuminate all crystals before time runs out."
    )

    # Frames advance only when an action is received.
    auto_advance = False

    # --- Constants ---
    # Game parameters
    GRID_WIDTH = 18
    GRID_HEIGHT = 12
    NUM_CRYSTALS = 15
    MAX_STEPS = 600

    # Visual parameters
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TILE_WIDTH_HALF = 20
    TILE_HEIGHT_HALF = 10
    CRYSTAL_HEIGHT = 20

    # Colors
    COLOR_BG = (25, 20, 35)
    COLOR_PLATE = (70, 70, 90)
    COLOR_PLATE_LIT_GLOW = (150, 200, 255)
    
    COLOR_CRYSTAL_UNLIT = (60, 60, 150)
    COLOR_CRYSTAL_LIT = (220, 255, 255)
    
    COLOR_SELECT_OUTLINE = (50, 200, 255)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 28)

        # Centering offset for the grid
        self.grid_offset_x = (self.SCREEN_WIDTH // 2)
        self.grid_offset_y = (self.SCREEN_HEIGHT // 2) - (self.GRID_HEIGHT * self.TILE_HEIGHT_HALF) + 40

        # State variables are initialized in reset()
        self.crystal_positions = []
        self.plate_positions = []
        self.selected_crystal_idx = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_move_trail = None

        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_move_trail = None

        # Procedurally generate the level
        all_coords = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.np_random.shuffle(all_coords)

        # Ensure we have enough unique positions
        num_items = self.NUM_CRYSTALS * 2
        if len(all_coords) < num_items:
            raise ValueError("Grid is too small for the number of crystals and plates.")

        self.plate_positions = [tuple(c) for c in all_coords[:self.NUM_CRYSTALS]]
        self.crystal_positions = [list(c) for c in all_coords[self.NUM_CRYSTALS:num_items]]

        self.selected_crystal_idx = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1
        shift_pressed = action[2] == 1

        reward = -0.01  # Cost for taking a step
        self.last_move_trail = None # Clear trail from previous step

        lit_before = self._count_lit_crystals()

        # --- Action Handling ---
        # 1. Selection Change (Space/Shift)
        if space_pressed:
            self.selected_crystal_idx = (self.selected_crystal_idx + 1) % self.NUM_CRYSTALS
            # Sound: UI_Select_Up.wav
        elif shift_pressed:
            self.selected_crystal_idx = (self.selected_crystal_idx - 1 + self.NUM_CRYSTALS) % self.NUM_CRYSTALS
            # Sound: UI_Select_Down.wav

        # 2. Movement (Arrows)
        if movement > 0:
            # 0=none, 1=up, 2=down, 3=left, 4=right
            # Grid directions are different from isometric screen directions
            dirs = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
            dx, dy = dirs[movement]

            start_pos = self.crystal_positions[self.selected_crystal_idx]
            current_pos = list(start_pos)
            
            # Push logic: slide until collision
            while True:
                next_pos = [current_pos[0] + dx, current_pos[1] + dy]
                
                # Check for wall collision
                if not (0 <= next_pos[0] < self.GRID_WIDTH and 0 <= next_pos[1] < self.GRID_HEIGHT):
                    # Sound: Crystal_Hit_Wall.wav
                    break

                # Check for other crystal collision
                is_blocked = False
                for i, pos in enumerate(self.crystal_positions):
                    if i != self.selected_crystal_idx and pos == next_pos:
                        is_blocked = True
                        # Sound: Crystal_Hit_Crystal.wav
                        break
                
                if is_blocked:
                    break

                current_pos = next_pos
            
            # If the crystal moved, update its position and create a trail
            if current_pos != start_pos:
                self.crystal_positions[self.selected_crystal_idx] = current_pos
                self.last_move_trail = (start_pos, current_pos)
                # Sound: Crystal_Slide.wav

        # --- Update Game State ---
        self.steps += 1
        
        lit_after = self._count_lit_crystals()
        newly_lit = lit_after - lit_before
        if newly_lit > 0:
            reward += newly_lit * 1.0
            # Sound: Crystal_Activate.wav
        
        self.score += reward
        terminated = self._check_termination()

        if terminated:
            self.game_over = True
            if lit_after == self.NUM_CRYSTALS:
                self.score += 100 # Win bonus
                reward += 100
                # Sound: Level_Complete.wav
            else: # Timeout
                self.score -= 100 # Loss penalty
                reward -= 100
                # Sound: Level_Fail.wav

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _check_termination(self):
        return self.steps >= self.MAX_STEPS or self._count_lit_crystals() == self.NUM_CRYSTALS

    def _count_lit_crystals(self):
        crystal_pos_set = {tuple(pos) for pos in self.crystal_positions}
        plate_pos_set = set(self.plate_positions)
        return len(crystal_pos_set.intersection(plate_pos_set))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lit_crystals": self._count_lit_crystals(),
            "time_left": self.MAX_STEPS - self.steps,
        }

    def _to_iso(self, x, y):
        iso_x = self.grid_offset_x + (x - y) * self.TILE_WIDTH_HALF
        iso_y = self.grid_offset_y + (x + y) * self.TILE_HEIGHT_HALF
        return int(iso_x), int(iso_y)

    def _draw_iso_cube(self, surface, iso_pos, color, height, outline_color=None, outline_width=2):
        x, y = iso_pos
        w, h = self.TILE_WIDTH_HALF, self.TILE_HEIGHT_HALF
        
        top_face = [(x, y - height), (x + w, y - height + h), (x, y - height + 2 * h), (x - w, y - height + h)]
        left_face = [(x - w, y - height + h), (x, y - height + 2 * h), (x, y + 2 * h), (x - w, y + h)]
        right_face = [(x + w, y - height + h), (x, y - height + 2 * h), (x, y + 2 * h), (x + w, y + h)]
        
        # Shading for 3D effect
        color_light = tuple(min(255, c + 30) for c in color)
        color_dark = tuple(max(0, c - 30) for c in color)
        
        pygame.gfxdraw.filled_polygon(surface, top_face, color_light)
        pygame.gfxdraw.filled_polygon(surface, left_face, color_dark)
        pygame.gfxdraw.filled_polygon(surface, right_face, color)

        # Draw outline if specified
        if outline_color:
            pygame.draw.polygon(surface, outline_color, top_face, outline_width)
            pygame.draw.polygon(surface, outline_color, left_face, outline_width)
            pygame.draw.polygon(surface, outline_color, right_face, outline_width)
    
    def _render_game(self):
        # Create a set of lit plate positions for quick lookup
        crystal_pos_set = {tuple(pos) for pos in self.crystal_positions}
        plate_pos_set = set(self.plate_positions)
        lit_plates = crystal_pos_set.intersection(plate_pos_set)

        # Draw pressure plates first
        for pos in self.plate_positions:
            iso_x, iso_y = self._to_iso(pos[0], pos[1])
            is_lit = pos in lit_plates
            
            plate_color = self.COLOR_PLATE
            if is_lit:
                # Draw a glowing effect for lit plates
                glow_radius = self.TILE_WIDTH_HALF + 5
                glow_center = (iso_x, iso_y + self.TILE_HEIGHT_HALF)
                
                # Simple radial gradient for glow
                for i in range(glow_radius, 0, -2):
                    alpha = int(100 * (1 - i / glow_radius))
                    color = (*self.COLOR_PLATE_LIT_GLOW, alpha)
                    pygame.gfxdraw.filled_circle(self.screen, glow_center[0], glow_center[1], i, color)

            pygame.gfxdraw.filled_circle(self.screen, iso_x, iso_y + self.TILE_HEIGHT_HALF, self.TILE_WIDTH_HALF - 2, plate_color)
            pygame.gfxdraw.aacircle(self.screen, iso_x, iso_y + self.TILE_HEIGHT_HALF, self.TILE_WIDTH_HALF - 2, plate_color)

        # Draw movement trail if it exists
        if self.last_move_trail:
            start_grid, end_grid = self.last_move_trail
            start_iso = self._to_iso(start_grid[0], start_grid[1])
            end_iso = self._to_iso(end_grid[0], end_grid[1])
            
            # Draw a fading line for the trail effect
            num_points = 10
            for i in range(num_points):
                t = i / (num_points - 1)
                x = int(start_iso[0] * (1 - t) + end_iso[0] * t)
                y = int(start_iso[1] * (1 - t) + end_iso[1] * t)
                alpha = int(150 * (1 - t)) # Fade out
                pygame.gfxdraw.filled_circle(self.screen, x, y, 3, (*self.COLOR_SELECT_OUTLINE, alpha))

        # Sort crystals by grid y then x for correct isometric rendering order
        sorted_crystals = sorted(enumerate(self.crystal_positions), key=lambda item: (item[1][1], item[1][0]))

        # Draw crystals
        for idx, pos in sorted_crystals:
            iso_pos = self._to_iso(pos[0], pos[1])
            is_lit = tuple(pos) in lit_plates
            is_selected = (idx == self.selected_crystal_idx)

            crystal_color = self.COLOR_CRYSTAL_LIT if is_lit else self.COLOR_CRYSTAL_UNLIT
            outline = self.COLOR_SELECT_OUTLINE if is_selected else None
            
            self._draw_iso_cube(self.screen, iso_pos, crystal_color, self.CRYSTAL_HEIGHT, outline_color=outline)

    def _render_ui(self):
        # Helper to draw text with a shadow
        def draw_text(text, font, color, pos, shadow_color):
            shadow_surface = font.render(text, True, shadow_color)
            text_surface = font.render(text, True, color)
            self.screen.blit(shadow_surface, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(text_surface, pos)

        # Lit Crystals Counter
        lit_count = self._count_lit_crystals()
        lit_text = f"LIT: {lit_count} / {self.NUM_CRYSTALS}"
        draw_text(lit_text, self.font_small, self.COLOR_TEXT, (20, 20), self.COLOR_TEXT_SHADOW)
        
        # Timer
        time_left = (self.MAX_STEPS - self.steps) / 10.0 # Assuming 10 steps/sec for display
        time_text = f"TIME: {time_left:.1f}"
        text_width = self.font_small.size(time_text)[0]
        draw_text(time_text, self.font_small, self.COLOR_TEXT, (self.SCREEN_WIDTH - text_width - 20, 20), self.COLOR_TEXT_SHADOW)

        # Game Over Message
        if self.game_over:
            is_win = lit_count == self.NUM_CRYSTALS
            msg = "COMPLETE" if is_win else "TIME UP"
            color = (180, 255, 180) if is_win else (255, 180, 180)
            
            text_width, text_height = self.font_large.size(msg)
            pos = ((self.SCREEN_WIDTH - text_width) // 2, (self.SCREEN_HEIGHT - text_height) // 2)
            draw_text(msg, self.font_large, color, pos, self.COLOR_TEXT_SHADOW)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a separate display for human play
    pygame.display.set_caption("Crystal Cavern")
    human_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    
    # Action state
    movement = 0 # 0: none, 1: up, 2: down, 3: left, 4: right
    space_held = 0
    shift_held = 0

    print("\n" + "="*30)
    print(f"GAME: {env.game_description}")
    print(f"CONTROLS: {env.user_guide}")
    print("="*30 + "\n")

    while not terminated:
        action_taken = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            
            # --- Key Down Events ---
            if event.type == pygame.KEYDOWN:
                action_taken = True
                if event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                elif event.key == pygame.K_SPACE:
                    space_held = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_held = 1
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                elif event.key == pygame.K_q: # Quit
                    terminated = True

        if action_taken:
            action = [movement, space_held, shift_held]
            obs, reward, term, trunc, info = env.step(action)
            terminated = term

            print(f"Step: {info['steps']}, Lit: {info['lit_crystals']}, Score: {info['score']:.2f}, Reward: {reward:.2f}")

            # Reset momentary actions
            movement = 0
            space_held = 0
            shift_held = 0

        # Render the observation to the human-facing screen
        # Need to transpose back from (H, W, C) to (W, H, C) for pygame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit frame rate for human play

    env.close()
    pygame.quit()