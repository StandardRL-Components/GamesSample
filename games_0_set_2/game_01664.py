
# Generated: 2025-08-27T17:52:26.272702
# Source Brief: brief_01664.md
# Brief Index: 1664

        
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

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a crystal, "
        "then move to an adjacent crystal and press Space again to swap them."
    )

    game_description = (
        "Swap adjacent crystals to create matches of 3 or more in this isometric "
        "puzzler. Clear the entire board before the timer runs out to win!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # --- Game Constants ---
        self.GRID_WIDTH = 8
        self.GRID_HEIGHT = 8
        self.NUM_CRYSTAL_TYPES = 3
        self.MAX_STEPS = 1800  # Equivalent to 60s at 30 "actions per second"
        self.ANIMATION_SPEED = 4  # Frames per animation phase

        # --- Colors ---
        self.COLOR_BG = (25, 20, 35)
        self.COLOR_GRID = (45, 40, 55)
        self.CRYSTAL_COLORS = {
            1: ((255, 50, 50), (200, 20, 20), (150, 10, 10)),   # Red
            2: ((50, 255, 50), (20, 200, 20), (10, 150, 10)),   # Green
            3: ((80, 80, 255), (40, 40, 200), (20, 20, 150)),   # Blue
        }
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_GOLD = (255, 215, 0)
        self.COLOR_UI_TEXT = (220, 220, 240)

        # --- Isometric Projection ---
        self.TILE_WIDTH_HALF = 22
        self.TILE_HEIGHT_HALF = 11
        self.TILE_DEPTH = 18
        self.ORIGIN_X = 640 // 2
        self.ORIGIN_Y = 90

        # --- Game State ---
        self.grid = None
        self.visual_offsets = None
        self.cursor_pos = None
        self.player_state = None
        self.selected_pos = None
        self.score = 0
        self.steps = 0
        self.steps_remaining = 0
        self.game_over = False
        self.prev_space_held = False
        self.particles = []
        self.animation_state = None
        self.animation_timer = 0
        self.matched_cells = set()
        self.crystals_to_fall = []
        self.total_crystals = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.score = 0
        self.steps = 0
        self.steps_remaining = self.MAX_STEPS
        self.game_over = False
        self.player_state = "IDLE"
        self.selected_pos = None
        self.cursor_pos = (self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2)
        self.prev_space_held = False
        self.particles = []
        self.animation_state = None
        self.animation_timer = 0

        self._generate_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        self.steps_remaining -= 1

        if self.animation_state:
            self._update_animation()
        else:
            reward = self._process_player_input(action)

        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.total_crystals == 0:
                reward += 100 # Win bonus
            else:
                reward -= 100 # Lose penalty

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _process_player_input(self, action):
        movement, space_held, _ = action
        space_press = space_held and not self.prev_space_held
        self.prev_space_held = bool(space_held)
        reward = 0

        # --- Cursor Movement ---
        r, c = self.cursor_pos
        if movement == 1: r -= 1  # Up
        elif movement == 2: r += 1  # Down
        elif movement == 3: c -= 1  # Left
        elif movement == 4: c += 1  # Right
        self.cursor_pos = (max(0, min(self.GRID_HEIGHT - 1, r)), max(0, min(self.GRID_WIDTH - 1, c)))

        # --- Action State Machine ---
        if space_press:
            if self.player_state == "IDLE":
                self.player_state = "SELECTED"
                self.selected_pos = self.cursor_pos
                # sfx: select_crystal
            elif self.player_state == "SELECTED":
                is_adjacent = abs(self.cursor_pos[0] - self.selected_pos[0]) + \
                              abs(self.cursor_pos[1] - self.selected_pos[1]) == 1
                if is_adjacent:
                    # sfx: swap_attempt
                    success, match_reward = self._attempt_swap(self.selected_pos, self.cursor_pos)
                    reward += 0.1 if success else -0.1
                    reward += match_reward
                # Reset selection regardless of outcome
                self.player_state = "IDLE"
                self.selected_pos = None
        return reward

    def _attempt_swap(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        
        # Perform swap
        self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]

        matches1 = self._find_matches_at(r1, c1)
        matches2 = self._find_matches_at(r2, c2)
        all_matches = matches1.union(matches2)

        if all_matches:
            self.matched_cells = all_matches
            self.animation_state = "MATCH_FLASH"
            self.animation_timer = self.ANIMATION_SPEED
            match_reward = 0
            
            # Group matches to calculate combo bonuses correctly
            grouped_matches = self._group_contiguous_matches(all_matches)
            for match_group in grouped_matches:
                n = len(match_group)
                match_reward += n  # +1 for each crystal
                if n > 3:
                    match_reward += (n - 3) * 2 # Combo bonus
            
            self.score += match_reward
            return True, match_reward
        else:
            # Swap back
            self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
            # sfx: invalid_swap
            return False, 0

    def _update_animation(self):
        self.animation_timer -= 1
        if self.animation_timer > 0:
            return

        if self.animation_state == "MATCH_FLASH":
            # sfx: match_success
            for r, c in self.matched_cells:
                self._spawn_particles(r, c)
                self.grid[r, c] = 0
                self.total_crystals -= 1
            self.animation_state = "FALL"
            self.animation_timer = self.ANIMATION_SPEED * 2
            self._prepare_fall_data()
            
        elif self.animation_state == "FALL":
            # sfx: crystals_land
            self._apply_gravity()
            self._refill_board()
            self.matched_cells.clear()
            
            # Check for chain reactions
            all_matches = self._find_all_matches()
            if all_matches:
                self.matched_cells = all_matches
                chain_reward = 0
                grouped_matches = self._group_contiguous_matches(all_matches)
                for match_group in grouped_matches:
                    n = len(match_group)
                    chain_reward += n * 1.5 # Chain reaction bonus
                    if n > 3:
                        chain_reward += (n - 3) * 3
                self.score += int(chain_reward)
                self.animation_state = "MATCH_FLASH"
                self.animation_timer = self.ANIMATION_SPEED
            else:
                self.animation_state = None
                if not self._check_for_possible_moves():
                    # sfx: board_reshuffle
                    self._reshuffle_board()


    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_crystals()
        self._render_cursor()
        self._update_and_render_particles()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                x, y = self._iso_to_screen(r, c)
                points = [
                    (x, y - self.TILE_HEIGHT_HALF),
                    (x + self.TILE_WIDTH_HALF, y),
                    (x, y + self.TILE_HEIGHT_HALF),
                    (x - self.TILE_WIDTH_HALF, y)
                ]
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_GRID)
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

    def _render_crystals(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                crystal_type = self.grid[r, c]
                if crystal_type == 0:
                    continue

                x, y = self._iso_to_screen(r, c)
                
                # Apply visual offsets for animations
                offset_x, offset_y = self.visual_offsets[r,c]
                y += offset_y

                scale = 1.0
                if self.animation_state == "MATCH_FLASH" and (r,c) in self.matched_cells:
                    # Pulsate matched crystals
                    scale = 1.0 + 0.3 * math.sin(self.animation_timer / self.ANIMATION_SPEED * math.pi)

                self._draw_iso_poly(
                    self.screen, x, y,
                    self.TILE_WIDTH_HALF * scale,
                    self.TILE_HEIGHT_HALF * scale,
                    self.TILE_DEPTH * scale,
                    self.CRYSTAL_COLORS[crystal_type]
                )

    def _render_cursor(self):
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        
        # Highlight under cursor
        r, c = self.cursor_pos
        x, y = self._iso_to_screen(r, c)
        cursor_color = self.COLOR_WHITE if self.player_state == "IDLE" else self.COLOR_GOLD
        alpha = int(100 + 100 * pulse)
        
        points = [
            (x, y - self.TILE_HEIGHT_HALF - 3),
            (x + self.TILE_WIDTH_HALF + 3, y),
            (x, y + self.TILE_HEIGHT_HALF + 3),
            (x - self.TILE_WIDTH_HALF - 3, y)
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, (*cursor_color, alpha))
        pygame.gfxdraw.aapolygon(self.screen, [(p[0]+1, p[1]) for p in points], (*cursor_color, alpha//2))


        # Highlight selected crystal
        if self.player_state == "SELECTED" and self.selected_pos:
            r, c = self.selected_pos
            x, y = self._iso_to_screen(r, c)
            points = [
                (x, y - self.TILE_HEIGHT_HALF),
                (x + self.TILE_WIDTH_HALF, y),
                (x, y + self.TILE_HEIGHT_HALF),
                (x - self.TILE_WIDTH_HALF, y)
            ]
            for i in range(3):
                offset_points = [(p[0], p[1]-i) for p in points]
                pygame.gfxdraw.aapolygon(self.screen, offset_points, (*self.COLOR_GOLD, 150 - i*40))

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))

        timer_text = self.font_main.render(f"TIME: {self.steps_remaining}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.screen.get_width() - timer_text.get_width() - 20, 10))
        
        crystals_text = self.font_small.render(f"Crystals Left: {self.total_crystals}", True, self.COLOR_UI_TEXT)
        self.screen.blit(crystals_text, (self.screen.get_width() // 2 - crystals_text.get_width() // 2, 15))

        if self.game_over:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "YOU WIN!" if self.total_crystals == 0 else "TIME UP!"
            end_text = self.font_main.render(msg, True, self.COLOR_GOLD)
            self.screen.blit(end_text, (320 - end_text.get_width() // 2, 200 - end_text.get_height() // 2))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- Helper Functions ---
    def _iso_to_screen(self, r, c):
        x = self.ORIGIN_X + (c - r) * self.TILE_WIDTH_HALF
        y = self.ORIGIN_Y + (c + r) * self.TILE_HEIGHT_HALF
        return int(x), int(y)

    def _draw_iso_poly(self, surface, x, y, w_half, h_half, depth, colors):
        top_color, side_color1, side_color2 = colors
        y_base = y + h_half
        
        # Right face
        points_right = [(x, y_base), (x + w_half, y), (x + w_half, y + depth), (x, y_base + depth)]
        pygame.gfxdraw.filled_polygon(surface, points_right, side_color1)
        pygame.gfxdraw.aapolygon(surface, points_right, side_color1)
        
        # Left face
        points_left = [(x, y_base), (x - w_half, y), (x - w_half, y + depth), (x, y_base + depth)]
        pygame.gfxdraw.filled_polygon(surface, points_left, side_color2)
        pygame.gfxdraw.aapolygon(surface, points_left, side_color2)

        # Top face
        points_top = [(x, y_base - depth), (x + w_half, y - depth + h_half), (x, y_base), (x - w_half, y - depth + h_half)]
        pygame.gfxdraw.filled_polygon(surface, points_top, top_color)
        pygame.gfxdraw.aapolygon(surface, points_top, top_color)

    def _generate_board(self):
        self.grid = self.np_random.integers(1, self.NUM_CRYSTAL_TYPES + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        self.visual_offsets = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH, 2), dtype=np.float32)
        
        # Ensure no initial matches and at least one possible move
        while self._find_all_matches() or not self._check_for_possible_moves():
            self.grid = self.np_random.integers(1, self.NUM_CRYSTAL_TYPES + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        
        self.total_crystals = self.GRID_WIDTH * self.GRID_HEIGHT

    def _find_matches_at(self, r, c):
        if r < 0 or r >= self.GRID_HEIGHT or c < 0 or c >= self.GRID_WIDTH:
            return set()
        
        color = self.grid[r, c]
        if color == 0:
            return set()

        # Horizontal
        h_matches = {(r, c)}
        for i in range(c - 1, -1, -1):
            if self.grid[r, i] == color: h_matches.add((r, i))
            else: break
        for i in range(c + 1, self.GRID_WIDTH):
            if self.grid[r, i] == color: h_matches.add((r, i))
            else: break
        
        # Vertical
        v_matches = {(r, c)}
        for i in range(r - 1, -1, -1):
            if self.grid[i, c] == color: v_matches.add((i, c))
            else: break
        for i in range(r + 1, self.GRID_HEIGHT):
            if self.grid[i, c] == color: v_matches.add((i, c))
            else: break
        
        found_matches = set()
        if len(h_matches) >= 3: found_matches.update(h_matches)
        if len(v_matches) >= 3: found_matches.update(v_matches)
        return found_matches

    def _find_all_matches(self):
        all_matches = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                all_matches.update(self._find_matches_at(r, c))
        return all_matches
    
    def _group_contiguous_matches(self, all_matches):
        if not all_matches:
            return []
        
        groups = []
        matches_left = set(all_matches)
        
        while matches_left:
            group = set()
            q = [matches_left.pop()]
            group.add(q[0])
            
            head = 0
            while head < len(q):
                r, c = q[head]
                head += 1
                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                    neighbor = (r + dr, c + dc)
                    if neighbor in matches_left:
                        matches_left.remove(neighbor)
                        group.add(neighbor)
                        q.append(neighbor)
            groups.append(group)
        return groups

    def _prepare_fall_data(self):
        self.visual_offsets.fill(0)
        for c in range(self.GRID_WIDTH):
            fall_dist = 0
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] == 0:
                    fall_dist += 1
                elif fall_dist > 0:
                    fall_pixels = fall_dist * (self.TILE_HEIGHT_HALF * 2)
                    self.visual_offsets[r,c,1] = -fall_pixels

    def _apply_gravity(self):
        self.visual_offsets.fill(0)
        for c in range(self.GRID_WIDTH):
            empty_r = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != empty_r:
                        self.grid[empty_r, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    empty_r -= 1

    def _refill_board(self):
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_CRYSTAL_TYPES + 1)
                    self.total_crystals += 1
                    fall_pixels = (r + 1) * (self.TILE_HEIGHT_HALF * 2)
                    self.visual_offsets[r,c,1] = -fall_pixels

    def _check_for_possible_moves(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                # Check swap right
                if c < self.GRID_WIDTH - 1:
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                    if self._find_matches_at(r,c) or self._find_matches_at(r,c+1):
                        self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                        return True
                    self.grid[r, c], self.grid[r, c+1] = self.grid[r, c+1], self.grid[r, c]
                # Check swap down
                if r < self.GRID_HEIGHT - 1:
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
                    if self._find_matches_at(r,c) or self._find_matches_at(r+1,c):
                        self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
                        return True
                    self.grid[r, c], self.grid[r+1, c] = self.grid[r+1, c], self.grid[r, c]
        return False
    
    def _reshuffle_board(self):
        flat_list = self.grid.flatten().tolist()
        self.np_random.shuffle(flat_list)
        self.grid = np.array(flat_list).reshape((self.GRID_HEIGHT, self.GRID_WIDTH))
        while self._find_all_matches() or not self._check_for_possible_moves():
            self.np_random.shuffle(flat_list)
            self.grid = np.array(flat_list).reshape((self.GRID_HEIGHT, self.GRID_WIDTH))

    def _spawn_particles(self, r, c):
        x, y = self._iso_to_screen(r, c)
        color = self.CRYSTAL_COLORS[self.grid[r, c]][0]
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed - 1)
            life = random.randint(15, 30)
            self.particles.append({'pos': [x, y], 'vel': vel, 'life': life, 'color': color})

    def _update_and_render_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'] = (p['vel'][0] * 0.98, p['vel'][1] + 0.1) # Damping and gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                size = max(0, p['life'] / 10)
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                pygame.draw.circle(self.screen, p['color'], pos, size)

    def _check_termination(self):
        return self.total_crystals == 0 or self.steps_remaining <= 0 or self.steps >= self.MAX_STEPS

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Interactive Gameplay Loop ---
    # This allows a human to play the game.
    # The agent's action is determined by keyboard input.
    
    pygame.display.set_caption("Crystal Cavern")
    screen = pygame.display.set_mode((640, 400))
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op
    
    while not done:
        # Map keyboard input to the MultiDiscrete action space
        move_action = 0 # none
        space_action = 0 # released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: move_action = 1
        elif keys[pygame.K_DOWN]: move_action = 2
        elif keys[pygame.K_LEFT]: move_action = 3
        elif keys[pygame.K_RIGHT]: move_action = 4
        
        if keys[pygame.K_SPACE]: space_action = 1

        # NOTE: In a real game, you'd only step on a key press event.
        # Here, we step every frame to keep the game responsive and allow
        # for animations to play out smoothly even with auto_advance=False.
        # The game logic itself only processes player input when not animating.
        
        # Only register a new action if the game is not in an animation sequence
        if not env.animation_state:
            action = np.array([move_action, space_action, 0])
        else:
            # During animation, send a no-op action but still step to advance animation
            action.fill(0)
            
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # This clock tick is for the human player display,
        # it doesn't affect the environment's internal step count.
        clock.tick(30)
        
    print(f"Game Over. Final Score: {info['score']}")
    pygame.quit()