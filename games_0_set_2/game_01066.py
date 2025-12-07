# Generated: 2025-08-27T15:46:38.205377
# Source Brief: brief_01066.md
# Brief Index: 1066

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move cursor. Press Space to swap the crystal "
        "at the cursor with the one in the direction you last moved."
    )

    game_description = (
        "An isometric match-3 puzzle game. Swap adjacent crystals to create lines of "
        "3 or more. Clear as many as you can before the 60-second timer runs out!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 8, 10
        self.NUM_COLORS = 4 # Red, Green, Blue, Yellow
        self.TIME_LIMIT_S = 60
        self.FPS = 30

        # Isometric projection constants
        self.TILE_W_HALF = 24
        self.TILE_H_HALF = 14
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 80

        # Colors
        self.COLOR_BG = pygame.Color("#2c3e50")
        self.COLOR_GRID = pygame.Color("#34495e")
        self.CRYSTAL_COLORS = [
            pygame.Color("#e74c3c"), # Red
            pygame.Color("#2ecc71"), # Green
            pygame.Color("#3498db"), # Blue
            pygame.Color("#f1c40f"), # Yellow
        ]
        self.COLOR_CURSOR = pygame.Color("#ecf0f1")
        self.COLOR_TEXT = pygame.Color("#ffffff")

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Game state variables (initialized in reset)
        self.grid = None
        self.cursor_pos = None
        self.last_move_direction = 0
        self.score = 0
        self.steps = 0
        self.time_remaining = 0
        self.animations = []
        self.particles = []
        self.game_over = False
        self.rng = None

        self.reset()
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            if self.rng is None:
                self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.TIME_LIMIT_S
        self.cursor_pos = (self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2)
        self.last_move_direction = 0 # 1=U, 2=D, 3=L, 4=R
        self.animations = []
        self.particles = []
        
        self._generate_board()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        
        # --- Update game state (animations, timer) ---
        self.time_remaining -= 1 / self.FPS
        self.steps += 1
        
        animation_reward = self._update_animations()
        reward += animation_reward

        is_board_busy = len(self.animations) > 0

        # --- Process player input ---
        if not is_board_busy:
            # Handle cursor movement
            if movement != 0:
                prev_pos = self.cursor_pos
                r, c = self.cursor_pos
                if movement == 1: r -= 1 # Up
                elif movement == 2: r += 1 # Down
                elif movement == 3: c -= 1 # Left
                elif movement == 4: c += 1 # Right
                
                if 0 <= r < self.GRID_HEIGHT and 0 <= c < self.GRID_WIDTH:
                    self.cursor_pos = (r, c)
                
                if self.cursor_pos != prev_pos:
                    self.last_move_direction = movement

            # Handle swap action
            if space_held and self.last_move_direction != 0:
                r1, c1 = self.cursor_pos
                r2, c2 = r1, c1
                if self.last_move_direction == 1: r2 -= 1
                elif self.last_move_direction == 2: r2 += 1
                elif self.last_move_direction == 3: c2 -= 1
                elif self.last_move_direction == 4: c2 += 1
                
                if 0 <= r2 < self.GRID_HEIGHT and 0 <= c2 < self.GRID_WIDTH:
                    # Check if this swap would cause a match
                    if self._check_swap_for_match(r1, c1, r2, c2):
                        self.animations.append({
                            "type": "swap", "p1": (r1, c1), "p2": (r2, c2), "progress": 0.0
                        })
                        # sfx: swap_start
                    else:
                        # Invalid swap, trigger shake animation
                        self.animations.append({
                            "type": "shake", "p1": (r1, c1), "p2": (r2, c2), "progress": 0.0
                        })
                        reward -= 0.1 # Small penalty for invalid move
                        # sfx: invalid_move
                self.last_move_direction = 0 # Consume the direction on action

        # --- Check for termination ---
        terminated = self.time_remaining <= 0
        if terminated:
            self.game_over = True
            
        return self._get_observation(), reward, terminated, False, self._get_info()

    # --- Core Game Logic ---

    def _generate_board(self):
        self.grid = self.rng.integers(1, self.NUM_COLORS + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
        while True:
            matches = self._find_matches(self.grid)
            if not matches:
                if self._find_possible_moves():
                    break
                else: # No matches and no possible moves, reshuffle
                    self.grid = self.rng.integers(1, self.NUM_COLORS + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            else: # Has matches, reshuffle
                for r, c in matches:
                    self.grid[r, c] = self.rng.integers(1, self.NUM_COLORS + 1)

    def _find_matches(self, grid):
        matches = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if grid[r,c] == 0: continue
                # Horizontal
                if c < self.GRID_WIDTH - 2 and grid[r,c] == grid[r,c+1] == grid[r,c+2]:
                    matches.add((r,c)); matches.add((r,c+1)); matches.add((r,c+2))
                # Vertical
                if r < self.GRID_HEIGHT - 2 and grid[r,c] == grid[r+1,c] == grid[r+2,c]:
                    matches.add((r,c)); matches.add((r+1,c)); matches.add((r+2,c))
        return matches

    def _check_swap_for_match(self, r1, c1, r2, c2):
        temp_grid = self.grid.copy()
        temp_grid[r1, c1], temp_grid[r2, c2] = temp_grid[r2, c2], temp_grid[r1, c1]
        return len(self._find_matches(temp_grid)) > 0

    def _find_possible_moves(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                # Check swap right
                if c < self.GRID_WIDTH - 1 and self._check_swap_for_match(r, c, r, c + 1):
                    return True
                # Check swap down
                if r < self.GRID_HEIGHT - 1 and self._check_swap_for_match(r, c, r + 1, c):
                    return True
        return False

    def _apply_gravity_and_refill(self):
        cols_to_refill = [0] * self.GRID_WIDTH
        for c in range(self.GRID_WIDTH):
            write_ptr = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != 0:
                    self.grid[write_ptr, c], self.grid[r, c] = self.grid[r, c], self.grid[write_ptr, c]
                    if write_ptr != r:
                        self.animations.append({"type": "fall", "from_r": r, "to_r": write_ptr, "c": c, "progress": 0.0})
                    write_ptr -= 1
            cols_to_refill[c] = write_ptr + 1

        for c in range(self.GRID_WIDTH):
            for r in range(cols_to_refill[c]):
                self.grid[r, c] = self.rng.integers(1, self.NUM_COLORS + 1)
                self.animations.append({"type": "fall", "from_r": r - cols_to_refill[c], "to_r": r, "c": c, "progress": 0.0})

    def _update_animations(self):
        reward = 0
        active_animations = []
        
        # Update particles
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # Gravity
            p["life"] -= 1

        for anim in self.animations:
            anim["progress"] += 0.1 # Speed of animations
            
            if anim["progress"] >= 1.0:
                # --- Animation complete, apply state change ---
                if anim["type"] == "swap":
                    r1, c1 = anim["p1"]
                    r2, c2 = anim["p2"]
                    self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
                    
                    matches = self._find_matches(self.grid)
                    if matches:
                        # sfx: match_success
                        num_cleared = len(matches)
                        reward += num_cleared * 1.0 # Reward per cleared crystal
                        self.score += num_cleared * 10
                        
                        for r, c in matches:
                            self._spawn_particles(r, c, self.grid[r,c])
                            self.grid[r,c] = 0
                        
                        self._apply_gravity_and_refill()
                    else: # Cascade check after fall
                        new_matches = self._find_matches(self.grid)
                        if new_matches:
                            # sfx: cascade_match
                            num_cleared = len(new_matches)
                            reward += num_cleared * 1.5 # Bonus for cascade
                            self.score += num_cleared * 15
                            for r, c in new_matches:
                                self._spawn_particles(r, c, self.grid[r,c])
                                self.grid[r,c] = 0
                            self._apply_gravity_and_refill()

                elif anim["type"] == "shake" or anim["type"] == "fall":
                    pass # No state change on completion, just visual
            else:
                active_animations.append(anim)
        
        # If board just became free, check for new cascades
        if not active_animations and any(a["type"] in ["swap", "fall"] for a in self.animations):
            matches = self._find_matches(self.grid)
            if matches:
                # sfx: cascade_match
                num_cleared = len(matches)
                reward += num_cleared * 1.5
                self.score += num_cleared * 15
                for r, c in matches:
                    self._spawn_particles(r, c, self.grid[r,c])
                    self.grid[r,c] = 0
                self._apply_gravity_and_refill()
            elif not self._find_possible_moves():
                # No moves left, reshuffle
                self.animations.append({"type": "reshuffle", "progress": 0.0})
                self._generate_board() # This is instant for now
                # sfx: board_reshuffle

        self.animations = active_animations
        return reward

    # --- Rendering ---

    def _get_observation(self):
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self.screen.fill(self.COLOR_BG)
        self._draw_cavern_background()
        self._draw_grid_lines()
        self._draw_crystals()
        self._draw_cursor()
        self._draw_particles()

    def _draw_cavern_background(self):
        # Draw some dark, textured dots for a cavern feel
        for i in range(150):
            x = self.rng.integers(0, self.WIDTH)
            y = self.rng.integers(0, self.HEIGHT)
            c = self.rng.integers(40, 55)
            pygame.gfxdraw.pixel(self.screen, x, y, (c,c,c))

    def _draw_grid_lines(self):
        for r in range(self.GRID_HEIGHT + 1):
            p1 = self._iso_to_screen(r, 0)
            p2 = self._iso_to_screen(r, self.GRID_WIDTH)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)
        for c in range(self.GRID_WIDTH + 1):
            p1 = self._iso_to_screen(0, c)
            p2 = self._iso_to_screen(self.GRID_HEIGHT, c)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1, p2)

    def _draw_crystals(self):
        animating_crystals = {}
        for anim in self.animations:
            if anim["type"] == "swap":
                animating_crystals[anim["p1"]] = {"target": anim["p2"], "progress": anim["progress"]}
                animating_crystals[anim["p2"]] = {"target": anim["p1"], "progress": anim["progress"]}
            elif anim["type"] == "shake":
                animating_crystals[anim["p1"]] = {"shake": True, "progress": anim["progress"]}
                animating_crystals[anim["p2"]] = {"shake": True, "progress": anim["progress"]}
            elif anim["type"] == "fall":
                animating_crystals[(anim["from_r"], anim["c"])] = {"fall_to": anim["to_r"], "progress": anim["progress"]}

        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                crystal_type = self.grid[r, c]
                if crystal_type == 0:
                    continue

                screen_pos = self._iso_to_screen(r, c)
                
                # Apply animations
                if (r, c) in animating_crystals:
                    anim_info = animating_crystals[(r, c)]
                    if "target" in anim_info:
                        target_pos = self._iso_to_screen(*anim_info["target"])
                        progress = anim_info["progress"]
                        screen_pos = (
                            int(screen_pos[0] + (target_pos[0] - screen_pos[0]) * progress),
                            int(screen_pos[1] + (target_pos[1] - screen_pos[1]) * progress)
                        )
                    elif "shake" in anim_info:
                        progress = anim_info["progress"]
                        offset = math.sin(progress * math.pi * 4) * 4
                        screen_pos = (screen_pos[0] + int(offset), screen_pos[1])
                    elif "fall_to" in anim_info:
                        from_pos = self._iso_to_screen(r, c)
                        to_pos = self._iso_to_screen(anim_info["fall_to"], c)
                        progress = anim_info["progress"]
                        screen_pos = (
                            int(from_pos[0] + (to_pos[0] - from_pos[0]) * progress),
                            int(from_pos[1] + (to_pos[1] - from_pos[1]) * progress)
                        )
                
                self._draw_iso_crystal(screen_pos, crystal_type)

    def _draw_iso_crystal(self, pos, crystal_type):
        x, y = pos
        color = self.CRYSTAL_COLORS[crystal_type - 1]
        
        # Shimmer effect
        shimmer_alpha = (math.sin(self.steps * 0.1 + x + y) * 0.5 + 0.5) * 50
        shimmer_color = (255, 255, 255, shimmer_alpha)
        
        top_face = [
            (x, y - self.TILE_H_HALF),
            (x + self.TILE_W_HALF, y),
            (x, y + self.TILE_H_HALF),
            (x - self.TILE_W_HALF, y)
        ]
        
        left_face = [
            (x - self.TILE_W_HALF, y),
            (x, y + self.TILE_H_HALF),
            (x, y + self.TILE_H_HALF + self.TILE_H_HALF),
            (x - self.TILE_W_HALF, y + self.TILE_H_HALF)
        ]
        
        right_face = [
            (x + self.TILE_W_HALF, y),
            (x, y + self.TILE_H_HALF),
            (x, y + self.TILE_H_HALF + self.TILE_H_HALF),
            (x + self.TILE_W_HALF, y + self.TILE_H_HALF)
        ]
        
        pygame.gfxdraw.filled_polygon(self.screen, left_face, color.lerp("black", 0.4))
        pygame.gfxdraw.aapolygon(self.screen, left_face, color.lerp("black", 0.4))
        
        pygame.gfxdraw.filled_polygon(self.screen, right_face, color.lerp("black", 0.2))
        pygame.gfxdraw.aapolygon(self.screen, right_face, color.lerp("black", 0.2))
        
        pygame.gfxdraw.filled_polygon(self.screen, top_face, color)
        pygame.gfxdraw.aapolygon(self.screen, top_face, color)

        # Draw shimmer on top face
        shimmer_surface = pygame.Surface((self.TILE_W_HALF*2, self.TILE_H_HALF*2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_polygon(shimmer_surface, [(p[0]-x+self.TILE_W_HALF, p[1]-y+self.TILE_H_HALF) for p in top_face], shimmer_color)
        self.screen.blit(shimmer_surface, (x - self.TILE_W_HALF, y - self.TILE_H_HALF))

    def _draw_cursor(self):
        r, c = self.cursor_pos
        x, y = self._iso_to_screen(r, c)
        
        points = [
            (x, y - self.TILE_H_HALF - 2),
            (x + self.TILE_W_HALF + 2, y),
            (x, y + self.TILE_H_HALF + 2),
            (x - self.TILE_W_HALF - 2, y)
        ]
        
        alpha = int((math.sin(self.steps * 0.2) * 0.5 + 0.5) * 255)
        color = self.COLOR_CURSOR
        
        # The color argument must be a 3- or 4-element tuple/list.
        # self.COLOR_CURSOR is a pygame.Color object which unpacks to (r, g, b, a).
        # The original code (*color, alpha) created a 5-element tuple, causing the error.
        # We create a new 4-element tuple with the desired alpha.
        rgba_color = (color.r, color.g, color.b, alpha)
        pygame.draw.lines(self.screen, rgba_color, True, points, 2)

    def _draw_particles(self):
        for p in self.particles:
            size = int(p["life"] / p["max_life"] * 5)
            if size > 0:
                pygame.draw.circle(self.screen, p["color"], p["pos"], size)

    def _spawn_particles(self, r, c, crystal_type):
        x, y = self._iso_to_screen(r,c)
        color = self.CRYSTAL_COLORS[crystal_type - 1]
        for _ in range(15): # Spawn 15 particles
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            self.particles.append({
                "pos": [x, y],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed - 2],
                "color": color,
                "life": self.rng.integers(15, 30),
                "max_life": 30
            })

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer bar
        timer_ratio = max(0, self.time_remaining / self.TIME_LIMIT_S)
        bar_width = (self.WIDTH - 20) * timer_ratio
        bar_color = self.COLOR_TEXT.lerp("red", 1 - timer_ratio)
        pygame.draw.rect(self.screen, self.COLOR_GRID, (10, self.HEIGHT - 25, self.WIDTH - 20, 15))
        pygame.draw.rect(self.screen, bar_color, (10, self.HEIGHT - 25, bar_width, 15))
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_main.render("Time's Up!", True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 - 20))
            self.screen.blit(end_text, text_rect)
            
            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            score_rect = final_score_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 + 20))
            self.screen.blit(final_score_text, score_rect)

    # --- Helpers ---

    def _iso_to_screen(self, r, c):
        x = self.ORIGIN_X + (c - r) * self.TILE_W_HALF
        y = self.ORIGIN_Y + (c + r) * self.TILE_H_HALF
        return int(x), int(y)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "is_animating": len(self.animations) > 0,
        }

    def validate_implementation(self):
        ''' Call this at the end of __init__ to verify implementation. '''
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

if __name__ == "__main__":
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Un-set the dummy driver to allow for display
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Crystal Cavern")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # --- Player Input ---
        move_action = 0 # 0=none
        space_action = 0 # 0=released
        
        # This is a polling-based input, so we check every frame
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: move_action = 1
        elif keys[pygame.K_DOWN]: move_action = 2
        elif keys[pygame.K_LEFT]: move_action = 3
        elif keys[pygame.K_RIGHT]: move_action = 4
        
        if keys[pygame.K_SPACE]: space_action = 1
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        # --- Environment Step ---
        action = [move_action, space_action, 0] # Shift is not used
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait for a moment before allowing reset
            pygame.time.wait(2000)
            obs, info = env.reset()

        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    pygame.quit()