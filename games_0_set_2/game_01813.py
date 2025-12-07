
# Generated: 2025-08-28T02:47:42.104022
# Source Brief: brief_01813.md
# Brief Index: 1813

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the cursor. Space to select a gem, "
        "then move the cursor to an adjacent empty tile and press Space again to move it. "
        "Shift to cancel a selection."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric gem-matching puzzle game. Move gems to adjacent empty spots "
        "to form lines of 3 or more of the same color. Clear the board before you run out of moves!"
    )

    # Should frames auto-advance or wait for user input?
    # Set to False because this is a turn-based puzzle game. State should only change
    # when an action is explicitly taken. The player needs time to think.
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 8, 8
    INITIAL_MOVES = 50
    MAX_STEPS = 2500

    # Colors
    COLOR_BG = (25, 30, 35)
    COLOR_GRID_LINE = (40, 45, 55)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_HIGHLIGHT = (255, 255, 255, 100)
    GEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 150, 50),  # Orange
        (200, 80, 255),  # Purple
        (80, 220, 220),  # Cyan
    ]

    # Isometric projection
    TILE_W_HALF = 32
    TILE_H_HALF = 16
    ISO_OFFSET_X = SCREEN_WIDTH // 2
    ISO_OFFSET_Y = 80

    # Animation timings (in steps/frames)
    ANIM_MOVE_DURATION = 8
    ANIM_MATCH_DURATION = 12
    ANIM_FALL_DURATION = 6

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Segoe UI, Arial", 24, bold=True)
        self.font_msg = pygame.font.SysFont("Segoe UI, Arial", 48, bold=True)
        
        # State variables are initialized in reset()
        self.grid = []
        self.cursor_pos = [0, 0]
        self.selected_gem_coord = None
        self.game_phase = "IDLE"
        self.animations = []
        self.particles = []
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_condition_met = False
        self.last_action = np.array([0, 0, 0])
        self.turn_reward = 0
        self.chain_reaction_level = 0
        
        self.reset()
        
        # Run validation check
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.moves_left = self.INITIAL_MOVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_condition_met = False
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_gem_coord = None
        self.game_phase = "IDLE"
        self.animations.clear()
        self.particles.clear()
        self.last_action = np.array([0, 0, 0])
        
        self._create_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = self.game_over
        self.steps += 1
        
        # --- Handle Input ---
        if self.game_phase == "IDLE":
            self._handle_input(action)
        
        # --- Update Game State Machine ---
        if self.game_phase.startswith("ANIMATING"):
            self._update_animations()
        elif self.game_phase == "PROCESSING":
            reward, terminated = self._process_board()

        # --- Check for Termination ---
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        if self.game_phase == "IDLE" and not terminated:
            # Check for win/loss only when the board is settled
            if self.win_condition_met:
                reward += 100
                terminated = True
                self.game_phase = "GAME_OVER"
            elif self.moves_left <= 0:
                reward -= 100
                terminated = True
                self.game_phase = "GAME_OVER"
        
        self.game_over = terminated
        self.last_action = action

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    # --- Core Logic ---

    def _handle_input(self, action):
        movement, space, shift = action
        space_press = space == 1 and self.last_action[1] == 0
        shift_press = shift == 1 and self.last_action[2] == 0

        # Cursor Movement
        if movement != 0:
            # Debounce movement
            if self.last_action[0] == 0:
                dx, dy = 0, 0
                if movement == 1: dy = -1  # Up
                elif movement == 2: dy = 1   # Down
                elif movement == 3: dx = -1  # Left
                elif movement == 4: dx = 1   # Right
                
                self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dx, 0, self.GRID_WIDTH - 1)
                self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dy, 0, self.GRID_HEIGHT - 1)

        # Cancel Selection
        if shift_press and self.selected_gem_coord:
            self.selected_gem_coord = None

        # Select / Move
        if space_press:
            cx, cy = self.cursor_pos
            if not self.selected_gem_coord:
                # Select a gem
                if self.grid[cy][cx] is not None:
                    self.selected_gem_coord = (cx, cy)
            else:
                # Try to move selected gem
                sx, sy = self.selected_gem_coord
                is_adjacent = abs(cx - sx) + abs(cy - sy) == 1
                is_empty_target = self.grid[cy][cx] is None

                if is_adjacent and is_empty_target:
                    self.moves_left -= 1
                    self.turn_reward = -0.1
                    self.chain_reaction_level = 0
                    
                    gem_to_move = self.grid[sy][sx]
                    self.grid[sy][sx] = None
                    
                    anim = {
                        "type": "MOVE", "gem": gem_to_move,
                        "start_coord": (sx, sy), "end_coord": (cx, cy),
                        "progress": 0, "duration": self.ANIM_MOVE_DURATION,
                        "on_complete": lambda: self._on_move_complete(gem_to_move, (cx, cy))
                    }
                    self.animations.append(anim)
                    self.game_phase = "ANIMATING_MOVE"
                    self.selected_gem_coord = None
                else:
                    # Invalid move, deselect
                    self.selected_gem_coord = None

    def _process_board(self):
        matches = self._find_matches()
        
        if matches:
            self.chain_reaction_level += 1
            num_gems_matched = len(matches)
            
            # Calculate reward
            match_reward = num_gems_matched * 1.0
            if num_gems_matched >= 4: match_reward += 5.0
            match_reward *= self.chain_reaction_level # Combo bonus
            
            if self.turn_reward == -0.1: self.turn_reward = 0
            self.turn_reward += match_reward
            
            # Animate and remove gems
            for x, y in matches:
                gem = self.grid[y][x]
                if gem:
                    # Sound: gem match
                    self._create_particles(self._world_to_screen(x, y), gem["color"])
                    anim = {
                        "type": "MATCH", "gem": gem, "coord": (x, y),
                        "progress": 0, "duration": self.ANIM_MATCH_DURATION,
                    }
                    self.animations.append(anim)
                    self.grid[y][x] = None
            
            self.game_phase = "ANIMATING_MATCH"
            return 0, False # Reward is given at end of turn
        else:
            # No more matches, end the turn
            final_reward = self.turn_reward
            self.score += final_reward
            self.game_phase = "IDLE"
            
            # Check for win condition
            if all(self.grid[y][x] is None for y in range(self.GRID_HEIGHT) for x in range(self.GRID_WIDTH)):
                self.win_condition_met = True
            
            return final_reward, False

    def _find_matches(self):
        to_remove = set()
        # Horizontal
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH - 2):
                gem1 = self.grid[y][x]
                if gem1:
                    match_group = [(x, y)]
                    for i in range(1, self.GRID_WIDTH - x):
                        gem_next = self.grid[y][x + i]
                        if gem_next and gem_next["color"] == gem1["color"]:
                            match_group.append((x + i, y))
                        else:
                            break
                    if len(match_group) >= 3:
                        for pos in match_group: to_remove.add(pos)
        # Vertical
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT - 2):
                gem1 = self.grid[y][x]
                if gem1:
                    match_group = [(x, y)]
                    for i in range(1, self.GRID_HEIGHT - y):
                        gem_next = self.grid[y + i][x]
                        if gem_next and gem_next["color"] == gem1["color"]:
                            match_group.append((x, y + i))
                        else:
                            break
                    if len(match_group) >= 3:
                        for pos in match_group: to_remove.add(pos)
        return list(to_remove)

    def _apply_gravity_and_refill(self):
        gems_fell = False
        # Gravity
        for x in range(self.GRID_WIDTH):
            empty_y = -1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[y][x] is None and empty_y == -1:
                    empty_y = y
                elif self.grid[y][x] is not None and empty_y != -1:
                    gem = self.grid[y][x]
                    self.grid[empty_y][x] = gem
                    self.grid[y][x] = None
                    
                    anim = {
                        "type": "FALL", "gem": gem,
                        "start_coord": (x, y), "end_coord": (x, empty_y),
                        "progress": 0, "duration": self.ANIM_FALL_DURATION,
                    }
                    self.animations.append(anim)
                    gems_fell = True
                    empty_y -= 1

        # Refill
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[y][x] is None:
                    new_gem = self._create_gem()
                    spawn_y = y - self.GRID_HEIGHT
                    self.grid[y][x] = new_gem
                    anim = {
                        "type": "FALL", "gem": new_gem,
                        "start_coord": (x, spawn_y), "end_coord": (x, y),
                        "progress": 0, "duration": self.ANIM_FALL_DURATION,
                    }
                    self.animations.append(anim)
                    gems_fell = True
        
        self.game_phase = "ANIMATING_FALL" if gems_fell else "PROCESSING"

    # --- Animation Callbacks ---
    def _on_move_complete(self, gem, new_coord):
        self.grid[new_coord[1]][new_coord[0]] = gem
        self.game_phase = "PROCESSING"

    def _on_match_anim_complete(self):
        self._apply_gravity_and_refill()

    def _on_fall_anim_complete(self):
        self.game_phase = "PROCESSING"

    # --- Board & Gem Creation ---
    def _create_board(self):
        self.grid = [[self._create_gem() for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        # Ensure no initial matches
        while self._find_matches():
             matches = self._find_matches()
             for x, y in matches:
                 self.grid[y][x] = self._create_gem(exclude_color=self.grid[y][x]["color"])
        # Ensure at least one empty space for movement
        if all(g is not None for row in self.grid for g in row):
             self.grid[self.np_random.integers(0, self.GRID_HEIGHT)][self.np_random.integers(0, self.GRID_WIDTH)] = None

    def _create_gem(self, exclude_color=None):
        available_colors = [c for c in self.GEM_COLORS if c != exclude_color]
        color = random.choice(available_colors)
        return {"color": color, "scale": 1.0, "alpha": 255}

    # --- Rendering ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_grid_background()
        self._render_cursor_and_highlights()
        
        # Draw gems not being animated
        animating_gems = [a["gem"] for a in self.animations]
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                gem = self.grid[y][x]
                if gem and gem not in animating_gems:
                    screen_pos = self._world_to_screen(x, y)
                    self._draw_iso_gem(self.screen, screen_pos, gem)

        # Draw animated gems and particles
        self._render_animations()
        self._render_particles()

    def _render_grid_background(self):
        for y in range(self.GRID_HEIGHT + 1):
            start = self._world_to_screen(0, y)
            end = self._world_to_screen(self.GRID_WIDTH, y)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, start, end, 1)
        for x in range(self.GRID_WIDTH + 1):
            start = self._world_to_screen(x, 0)
            end = self._world_to_screen(x, self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, start, end, 1)
            
    def _render_cursor_and_highlights(self):
        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_poly = self._get_iso_tile_poly(cx, cy)
        pygame.draw.polygon(self.screen, self.COLOR_CURSOR, cursor_poly, 3)

        # Draw selected gem highlight
        if self.selected_gem_coord:
            sx, sy = self.selected_gem_coord
            select_poly = self._get_iso_tile_poly(sx, sy)
            
            highlight_surf = pygame.Surface((self.TILE_W_HALF*2, self.TILE_H_HALF*2), pygame.SRCALPHA)
            pygame.draw.polygon(highlight_surf, self.COLOR_HIGHLIGHT, [ (p[0] - select_poly[0][0] + self.TILE_W_HALF, p[1] - select_poly[0][1]) for p in select_poly])
            self.screen.blit(highlight_surf, (select_poly[0][0] - self.TILE_W_HALF, select_poly[0][1]))


    def _draw_iso_gem(self, surface, pos, gem):
        if not gem: return
        
        size_w = self.TILE_W_HALF * 0.85 * gem["scale"]
        size_h = self.TILE_H_HALF * 0.85 * gem["scale"]
        
        color = gem["color"]
        light_color = tuple(min(255, c + 60) for c in color)
        dark_color = tuple(max(0, c - 60) for c in color)
        
        px, py = int(pos[0]), int(pos[1])
        
        # Points for an octagon-like shape
        top_face = [
            (px, py - size_h * 0.5),
            (px + size_w * 0.5, py),
            (px, py + size_h * 0.5),
            (px - size_w * 0.5, py)
        ]
        
        left_face = [
            (px - size_w * 0.5, py),
            (px, py + size_h * 0.5),
            (px, py + size_h * 1.2),
            (px - size_w * 0.5, py + size_h * 0.7)
        ]
        
        right_face = [
            (px + size_w * 0.5, py),
            (px, py + size_h * 0.5),
            (px, py + size_h * 1.2),
            (px + size_w * 0.5, py + size_h * 0.7)
        ]

        if gem["alpha"] < 255:
            # This is slow, but necessary for per-gem alpha.
            temp_surf = pygame.Surface((size_w * 2, size_h * 3), pygame.SRCALPHA)
            temp_surf.set_alpha(gem["alpha"])
            offset_x, offset_y = px - size_w, py - size_h
            
            pygame.gfxdraw.filled_polygon(temp_surf, [(p[0]-offset_x, p[1]-offset_y) for p in left_face], dark_color)
            pygame.gfxdraw.filled_polygon(temp_surf, [(p[0]-offset_x, p[1]-offset_y) for p in right_face], dark_color)
            pygame.gfxdraw.filled_polygon(temp_surf, [(p[0]-offset_x, p[1]-offset_y) for p in top_face], color)
            pygame.gfxdraw.filled_polygon(temp_surf, [(p[0]-offset_x, p[1]-offset_y) for p in [(top_face[0]), (top_face[1]), (top_face[3])]], light_color)
            surface.blit(temp_surf, (offset_x, offset_y))
        else:
            pygame.gfxdraw.filled_polygon(surface, left_face, dark_color)
            pygame.gfxdraw.filled_polygon(surface, right_face, dark_color)
            pygame.gfxdraw.filled_polygon(surface, top_face, color)
            pygame.gfxdraw.filled_polygon(surface, [(top_face[0]),(top_face[1]),(top_face[3])], light_color)

    def _render_ui(self):
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, (240, 240, 240))
        moves_text = self.font_ui.render(f"Moves: {self.moves_left}", True, (240, 240, 240))
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 10, 10))
        
        if self.game_over:
            msg = "YOU WIN!" if self.win_condition_met else "GAME OVER"
            color = (100, 255, 100) if self.win_condition_met else (255, 100, 100)
            msg_surf = self.font_msg.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            # Add a backing for readability
            bg_rect = msg_rect.inflate(20, 20)
            bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surf.fill((0, 0, 0, 180))
            self.screen.blit(bg_surf, bg_rect)
            self.screen.blit(msg_surf, msg_rect)

    # --- Animation & Effects ---
    def _update_animations(self):
        if not self.animations: return
        
        completed_anims = []
        for anim in self.animations:
            anim["progress"] += 1
            if anim["progress"] >= anim["duration"]:
                completed_anims.append(anim)
        
        if completed_anims:
            self.animations = [a for a in self.animations if a not in completed_anims]
            
            # Group callbacks by type to avoid multiple phase changes
            callbacks = {"MOVE": None, "MATCH": None, "FALL": None}
            for anim in completed_anims:
                if "on_complete" in anim:
                    callbacks[anim["type"]] = anim["on_complete"]
            
            if callbacks["MOVE"]: callbacks["MOVE"]()
            elif callbacks["MATCH"]: self._on_match_anim_complete()
            elif callbacks["FALL"]: self._on_fall_anim_complete()

            if not self.animations:
                # If a callback didn't set a new phase, something is wrong, but we can recover.
                if self.game_phase.startswith("ANIMATING"):
                    self.game_phase = "PROCESSING"

    def _render_animations(self):
        for anim in self.animations:
            p = anim["progress"] / anim["duration"]
            
            if anim["type"] == "MOVE" or anim["type"] == "FALL":
                start_pos = self._world_to_screen(*anim["start_coord"])
                end_pos = self._world_to_screen(*anim["end_coord"])
                # Ease out
                eased_p = 1 - (1 - p) ** 2
                curr_pos = (
                    start_pos[0] + (end_pos[0] - start_pos[0]) * eased_p,
                    start_pos[1] + (end_pos[1] - start_pos[1]) * eased_p,
                )
                self._draw_iso_gem(self.screen, curr_pos, anim["gem"])
            elif anim["type"] == "MATCH":
                # Flash and shrink
                pos = self._world_to_screen(*anim["coord"])
                anim["gem"]["scale"] = 1.0 - p
                anim["gem"]["alpha"] = 255 * (1.0 - p)
                self._draw_iso_gem(self.screen, pos, anim["gem"])

    def _render_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            p["vel"][1] += 0.1 # gravity
            
            if p["life"] <= 0:
                self.particles.remove(p)
                continue
            
            radius = int(p["radius"] * (p["life"] / p["lifespan"]))
            if radius > 0:
                pygame.draw.circle(self.screen, p["color"], [int(x) for x in p["pos"]], radius)

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "color": color,
                "radius": self.np_random.integers(2, 5),
                "life": lifespan,
                "lifespan": lifespan
            })

    # --- Helper Methods ---
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    def _world_to_screen(self, x, y):
        px = self.ISO_OFFSET_X + (x - y) * self.TILE_W_HALF
        py = self.ISO_OFFSET_Y + (x + y) * self.TILE_H_HALF
        return (px, py)
        
    def _get_iso_tile_poly(self, x, y):
        px, py = self._world_to_screen(x, y)
        return [
            (px, py - self.TILE_H_HALF),
            (px + self.TILE_W_HALF, py),
            (px, py + self.TILE_H_HALF),
            (px - self.TILE_W_HALF, py),
        ]

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("✓ Running implementation validation...")
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