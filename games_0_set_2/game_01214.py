
# Generated: 2025-08-27T16:24:09.965985
# Source Brief: brief_01214.md
# Brief Index: 1214

        
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
        "Controls: Arrow keys to select a crystal. Hold SPACE and press an arrow key to push the selected crystal."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric puzzle game. Push crystals onto pressure plates to light them all up within the move limit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 14
    GRID_HEIGHT = 10
    NUM_CRYSTALS = 20
    MAX_MOVES = 50

    # Visuals
    TILE_WIDTH = 48
    TILE_HEIGHT = 24
    CRYSTAL_Z_HEIGHT = 30

    COLOR_BG = (25, 25, 40)
    COLOR_WALL = (40, 40, 60)
    COLOR_PLATE = (80, 80, 90)
    COLOR_PLATE_ACTIVE = (120, 180, 255)

    COLOR_CRYSTAL_DARK = (50, 50, 120)
    COLOR_CRYSTAL_LIT = (100, 180, 255)
    COLOR_CRYSTAL_FACE = (80, 80, 180)
    COLOR_CRYSTAL_FACE_LIT = (150, 220, 255)
    COLOR_CRYSTAL_TOP = (100, 100, 220)
    COLOR_CRYSTAL_TOP_LIT = (200, 240, 255)

    COLOR_SELECT_OUTLINE = (255, 255, 0)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 15)


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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.origin_x = self.SCREEN_WIDTH // 2
        self.origin_y = 80

        self.np_random = None

        # Game state variables are initialized in reset()
        self.crystals = []
        self.plates = []
        self.lit_mask = []
        self.selected_crystal_idx = 0
        self.moves_remaining = 0
        self.score = 0
        self.game_over = False
        self.last_reward = 0

        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()

        self._generate_level()
        self.selected_crystal_idx = 0
        self.moves_remaining = self.MAX_MOVES
        self.score = 0
        self.game_over = False
        self.last_reward = 0
        self._update_lit_state()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        is_push_action = space_held and movement > 0
        is_select_action = not space_held and movement > 0

        if is_push_action:
            # --- PUSH LOGIC (A REAL MOVE) ---
            self.moves_remaining -= 1
            # sfx: crystal_push_start.wav

            old_lit_count = sum(self.lit_mask)

            direction_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # up, down, left, right
            dx, dy = direction_map[movement]

            crystal_to_push = self.crystals[self.selected_crystal_idx]
            current_pos = list(crystal_to_push)
            
            # Slide until obstacle
            while True:
                next_pos = [current_pos[0] + dx, current_pos[1] + dy]
                
                # Check wall collision
                if not (0 <= next_pos[0] < self.GRID_WIDTH and 0 <= next_pos[1] < self.GRID_HEIGHT):
                    # sfx: crystal_hit_wall.wav
                    break 
                
                # Check other crystal collision
                if any(c == next_pos for i, c in enumerate(self.crystals) if i != self.selected_crystal_idx):
                    # sfx: crystal_hit_crystal.wav
                    break
                
                current_pos = next_pos

            self.crystals[self.selected_crystal_idx] = current_pos
            
            self._update_lit_state()
            new_lit_count = sum(self.lit_mask)
            
            # Calculate reward
            reward -= 0.2  # Cost for moving
            if new_lit_count > old_lit_count:
                reward += (new_lit_count - old_lit_count) * 5.0
                # sfx: new_crystal_lit.wav
            
            self.score += reward
            self.last_reward = reward

        elif is_select_action:
            # --- SELECT LOGIC (NOT A MOVE) ---
            # sfx: select_tick.wav
            if movement in [1, 4]: # Up / Right
                self.selected_crystal_idx = (self.selected_crystal_idx + 1) % self.NUM_CRYSTALS
            elif movement in [2, 3]: # Down / Left
                self.selected_crystal_idx = (self.selected_crystal_idx - 1 + self.NUM_CRYSTALS) % self.NUM_CRYSTALS
            self.last_reward = 0


        terminated = self._check_termination()
        if terminated and not self.game_over:
            if sum(self.lit_mask) == self.NUM_CRYSTALS:
                reward += 100 # Win bonus
                # sfx: victory_fanfare.wav
            else:
                reward -= 10 # Loss penalty
                # sfx: game_over_sound.wav
            self.score += reward
            self.last_reward = reward
            self.game_over = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _check_termination(self):
        return self.moves_remaining <= 0 or sum(self.lit_mask) == self.NUM_CRYSTALS

    def _update_lit_state(self):
        crystal_positions = {tuple(c) for c in self.crystals}
        self.lit_mask = []
        for p_pos in self.plates:
            is_on_plate = tuple(p_pos) in crystal_positions
            self.lit_mask.append(is_on_plate)

    def _generate_level(self):
        self.plates = []
        self.crystals = []
        occupied_coords = set()

        # 1. Place plates
        for _ in range(self.NUM_CRYSTALS):
            while True:
                pos = (
                    self.np_random.integers(0, self.GRID_WIDTH),
                    self.np_random.integers(0, self.GRID_HEIGHT)
                )
                if pos not in occupied_coords:
                    self.plates.append(list(pos))
                    occupied_coords.add(pos)
                    break
        
        # 2. Place crystals on a different set of coords
        crystal_coords = set()
        for _ in range(self.NUM_CRYSTALS):
            while True:
                pos = (
                    self.np_random.integers(0, self.GRID_WIDTH),
                    self.np_random.integers(0, self.GRID_HEIGHT)
                )
                if pos not in occupied_coords and pos not in crystal_coords:
                    self.crystals.append(list(pos))
                    crystal_coords.add(pos)
                    break

    def _iso_to_screen(self, x, y):
        screen_x = self.origin_x + (x - y) * (self.TILE_WIDTH / 2)
        screen_y = self.origin_y + (x + y) * (self.TILE_HEIGHT / 2)
        return int(screen_x), int(screen_y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Combine all items and sort by grid position for correct draw order
        items = []
        for i, pos in enumerate(self.plates):
            items.append({'type': 'plate', 'pos': pos, 'is_lit': self.lit_mask[i]})
        for i, pos in enumerate(self.crystals):
            is_lit = False
            for p_idx, p_pos in enumerate(self.plates):
                if p_pos == pos and self.lit_mask[p_idx]:
                    is_lit = True
                    break
            items.append({'type': 'crystal', 'pos': pos, 'is_lit': is_lit, 'is_selected': i == self.selected_crystal_idx})

        items.sort(key=lambda item: (item['pos'][1], item['pos'][0]))

        for item in items:
            if item['type'] == 'plate':
                self._draw_plate(item['pos'], item['is_lit'])
            elif item['type'] == 'crystal':
                self._draw_crystal(item['pos'], item['is_lit'], item['is_selected'])

    def _draw_plate(self, pos, is_active):
        screen_pos = self._iso_to_screen(pos[0], pos[1])
        color = self.COLOR_PLATE_ACTIVE if is_active else self.COLOR_PLATE
        pygame.gfxdraw.filled_ellipse(self.screen, screen_pos[0], screen_pos[1], self.TILE_WIDTH // 2 - 4, self.TILE_HEIGHT // 2 - 2, color)
        pygame.gfxdraw.aaellipse(self.screen, screen_pos[0], screen_pos[1], self.TILE_WIDTH // 2 - 4, self.TILE_HEIGHT // 2 - 2, color)

    def _draw_crystal(self, pos, is_lit, is_selected):
        sx, sy = self._iso_to_screen(pos[0], pos[1])
        z = self.CRYSTAL_Z_HEIGHT
        w = self.TILE_WIDTH / 2
        h = self.TILE_HEIGHT / 2

        # Points for the cube
        p_top_front = (sx, sy - z)
        p_top_right = (sx + w, sy - h - z)
        p_top_back = (sx, sy - 2 * h - z)
        p_top_left = (sx - w, sy - h - z)
        
        p_bottom_front = (sx, sy)
        p_bottom_right = (sx + w, sy - h)
        p_bottom_left = (sx - w, sy - h)

        # Colors
        top_color = self.COLOR_CRYSTAL_TOP_LIT if is_lit else self.COLOR_CRYSTAL_TOP
        face_color = self.COLOR_CRYSTAL_FACE_LIT if is_lit else self.COLOR_CRYSTAL_FACE
        dark_color = self.COLOR_CRYSTAL_LIT if is_lit else self.COLOR_CRYSTAL_DARK
        
        # Glow effect for lit crystals
        if is_lit:
            glow_radius = int(self.TILE_WIDTH * 1.2)
            glow_center = (sx, sy - z // 2)
            temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (150, 200, 255, 50), (glow_radius, glow_radius), glow_radius)
            pygame.draw.circle(temp_surf, (180, 220, 255, 30), (glow_radius, glow_radius), int(glow_radius * 0.7))
            self.screen.blit(temp_surf, (glow_center[0] - glow_radius, glow_center[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw faces
        pygame.draw.polygon(self.screen, top_color, [p_top_left, p_top_front, p_top_right, p_top_back]) # Top
        pygame.draw.polygon(self.screen, face_color, [p_bottom_front, p_bottom_left, p_top_left, p_top_front]) # Left face
        pygame.draw.polygon(self.screen, dark_color, [p_bottom_front, p_bottom_right, p_top_right, p_top_front]) # Right face
        
        # Outline
        outline_points = [p_top_left, p_top_front, p_top_right, p_bottom_right, p_bottom_front, p_bottom_left, p_top_left]
        pygame.draw.lines(self.screen, (0,0,0,50), False, outline_points, 1)

        # Selection highlight
        if is_selected:
            pygame.draw.polygon(self.screen, self.COLOR_SELECT_OUTLINE, [p_top_left, p_top_front, p_top_right, p_top_back], 2)
            pygame.draw.lines(self.screen, self.COLOR_SELECT_OUTLINE, False, [p_bottom_left, p_top_left, p_top_front, p_top_right, p_bottom_right], 2)

    def _render_text(self, text, font, x, y, color, shadow_color):
        text_surf = font.render(text, True, shadow_color)
        self.screen.blit(text_surf, (x + 2, y + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, (x, y))

    def _render_ui(self):
        # Lit Crystals Counter
        lit_count = sum(self.lit_mask)
        lit_text = f"Lit: {lit_count}/{self.NUM_CRYSTALS}"
        self._render_text(lit_text, self.font_large, 10, 10, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        # Moves Remaining Counter
        moves_text = f"Moves: {self.moves_remaining}"
        text_surf = self.font_large.render(moves_text, True, self.COLOR_TEXT)
        self._render_text(moves_text, self.font_large, self.SCREEN_WIDTH - text_surf.get_width() - 10, 10, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        # Game Over Message
        if self.game_over:
            if lit_count == self.NUM_CRYSTALS:
                msg = "SUCCESS!"
                color = (150, 255, 150)
            else:
                msg = "OUT OF MOVES"
                color = (255, 150, 150)
            
            self._render_text(msg, self.font_large, self.SCREEN_WIDTH//2 - self.font_large.size(msg)[0]//2, self.SCREEN_HEIGHT - 50, color, self.COLOR_TEXT_SHADOW)


    def _get_info(self):
        return {
            "score": self.score,
            "moves_remaining": self.moves_remaining,
            "lit_crystals": sum(self.lit_mask),
            "last_reward": self.last_reward,
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        self.reset()
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


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Crystal Cavern")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    terminated = False
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        action = [0, 0, 0] # no-op, release, release
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False
                print("--- GAME RESET ---")
        
        if not terminated:
            keys = pygame.key.get_pressed()
            
            # Movement keys for selection/push
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4

            # Modifier keys
            if keys[pygame.K_SPACE]:
                action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1

            # Since it's turn-based, we only step if an action is taken
            if action[0] > 0:
                obs, reward, terminated, truncated, info = env.step(action)
                if reward != 0:
                    print(f"Move: {env.MAX_MOVES - info['moves_remaining']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Lit: {info['lit_crystals']}")
                if terminated:
                    print(f"--- GAME OVER --- Final Score: {info['score']:.2f}")

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # The game is not auto-advancing, so we wait for player input
        # A small delay prevents the loop from running too fast
        pygame.time.wait(30)

    env.close()