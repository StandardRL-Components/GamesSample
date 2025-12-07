
# Generated: 2025-08-27T21:11:28.024529
# Source Brief: brief_02706.md
# Brief Index: 2706

        
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
        "Controls: Arrow keys to move the cursor. Space to flip a card."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced memory match game. Race against the clock to find all 8 pairs of symbols."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GRID_SIZE = 4
        self.TIME_LIMIT_SECONDS = 60

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (50, 60, 70)
        self.COLOR_CARD_HIDDEN = (70, 80, 90)
        self.COLOR_CARD_REVEALED = (210, 220, 230)
        self.COLOR_CARD_MATCHED = (40, 50, 60)
        self.COLOR_SYMBOL = (20, 30, 40)
        self.COLOR_CURSOR = (255, 180, 0)
        self.COLOR_MATCH = (0, 255, 120)
        self.COLOR_MISMATCH = (255, 80, 80)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TIMER_WARN = (255, 200, 0)
        self.COLOR_TIMER_DANGER = (255, 80, 80)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        
        # Initialize state variables
        self.grid = []
        self.cursor_pos = [0, 0]
        self.selected_cards = []
        self.mismatch_cooldown = 0
        self.last_mismatched_pair = []
        self.prev_space_held = False
        self.steps = 0
        self.score = 0
        self.matches_found = 0
        self.timer = 0
        self.game_over = False
        self.win = False

        self.reset()
        
        # Validate implementation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.matches_found = 0
        self.game_over = False
        self.win = False
        self.timer = self.TIME_LIMIT_SECONDS * self.FPS
        
        self.cursor_pos = [0, 0]
        self.selected_cards = []
        self.mismatch_cooldown = 0
        self.last_mismatched_pair = []
        self.prev_space_held = False

        self._generate_grid()
        
        return self._get_observation(), self._get_info()

    def _generate_grid(self):
        self.grid = []
        symbols = list(range(8)) * 2
        self.np_random.shuffle(symbols)

        padding = 10
        top_offset = 60
        grid_area_size = min(self.WIDTH, self.HEIGHT - top_offset)
        cell_size = (grid_area_size - padding * (self.GRID_SIZE + 1)) / self.GRID_SIZE
        start_x = (self.WIDTH - (cell_size * self.GRID_SIZE + padding * (self.GRID_SIZE - 1))) / 2
        
        for r in range(self.GRID_SIZE):
            row = []
            for c in range(self.GRID_SIZE):
                x = start_x + c * (cell_size + padding)
                y = top_offset + r * (cell_size + padding)
                card = {
                    "symbol": symbols.pop(),
                    "state": "hidden",  # hidden, revealed, matched
                    "rect": pygame.Rect(x, y, cell_size, cell_size),
                    "flip_progress": 0.0, # 0.0 is hidden, 1.0 is revealed
                    "known": False,
                    "match_glow": 0.0
                }
                row.append(card)
            self.grid.append(row)

    def step(self, action):
        reward = 0
        self.game_over = self.timer <= 0 or self.matches_found == 8
        if self.game_over:
            if self.matches_found == 8 and not self.win:
                reward = 100.0 # Big win bonus
                self.win = True
            return (
                self._get_observation(),
                reward,
                True,
                False,
                self._get_info(),
            )

        self.steps += 1
        self.timer -= 1

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # Handle cooldowns and animations
        self._update_animations()
        if self.mismatch_cooldown > 0:
            self.mismatch_cooldown -= 1
            if self.mismatch_cooldown == 0:
                # # SFX: Card flip back
                r1, c1 = self.last_mismatched_pair[0]
                r2, c2 = self.last_mismatched_pair[1]
                self.grid[r1][c1]["state"] = "hidden"
                self.grid[r2][c2]["state"] = "hidden"
                self.selected_cards.clear()

        # Process actions if not on cooldown
        if self.mismatch_cooldown == 0:
            self._handle_input(movement, space_held)
            reward += self._process_selection()

        self.prev_space_held = space_held
        
        terminated = self.timer <= 0 or self.matches_found == 8
        if terminated and self.matches_found == 8 and not self.win:
             reward = 100.0 # Big win bonus
             self.win = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _update_animations(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                card = self.grid[r][c]
                # Flip animation
                if card["state"] in ["revealed", "hidden"] and self.mismatch_cooldown == 0:
                    target_flip = 1.0 if card["state"] == "revealed" else 0.0
                    card["flip_progress"] += (target_flip - card["flip_progress"]) * 0.2
                # Match glow animation
                if card["state"] == "matched":
                    card["match_glow"] += (1.0 - card["match_glow"]) * 0.1
                else:
                    card["match_glow"] = 0.0

    def _handle_input(self, movement, space_held):
        # Move cursor
        if movement == 1: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 2: self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)
        elif movement == 3: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 4: self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)
        
        # Select card
        if space_held and not self.prev_space_held:
            r, c = self.cursor_pos
            card = self.grid[r][c]
            if card["state"] == "hidden" and len(self.selected_cards) < 2:
                # # SFX: Card flip
                card["state"] = "revealed"
                self.selected_cards.append((r, c))

    def _process_selection(self):
        reward = 0
        if len(self.selected_cards) == 1:
            r, c = self.selected_cards[0]
            card = self.grid[r][c]
            if not card["known"]:
                reward += 0.1
                card["known"] = True
            else:
                reward -= 0.01

        if len(self.selected_cards) == 2:
            r1, c1 = self.selected_cards[0]
            r2, c2 = self.selected_cards[1]
            card1 = self.grid[r1][c1]
            card2 = self.grid[r2][c2]

            if not card2["known"]:
                reward += 0.1
                card2["known"] = True
            else:
                reward -= 0.01

            if card1["symbol"] == card2["symbol"]:
                # Match
                # # SFX: Match success
                card1["state"] = "matched"
                card2["state"] = "matched"
                self.matches_found += 1
                self.score += 50
                reward += 10.0
                self.selected_cards.clear()
            else:
                # Mismatch
                # # SFX: Mismatch error
                self.mismatch_cooldown = int(self.FPS * 0.75)
                self.last_mismatched_pair = [(r1, c1), (r2, c2)]
                self.score -= 5
                reward -= 1.0
        
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                card = self.grid[r][c]
                rect = card["rect"]
                
                # Card shadow
                shadow_rect = rect.move(3, 3)
                pygame.draw.rect(self.screen, (0,0,0,50), shadow_rect, border_radius=8)

                # Card itself
                progress = card["flip_progress"]
                is_flipping = abs(progress - 0.5) < 0.5
                
                if is_flipping:
                    # Animate flip
                    anim_rect = rect.copy()
                    anim_rect.width = int(rect.width * abs(1 - 2 * progress))
                    anim_rect.center = rect.center
                    
                    if progress > 0.5: # Revealing face
                        pygame.draw.rect(self.screen, self.COLOR_CARD_REVEALED, anim_rect, border_radius=8)
                        if anim_rect.width > 20:
                             self._draw_symbol(card["symbol"], anim_rect)
                    else: # Revealing back
                        pygame.draw.rect(self.screen, self.COLOR_CARD_HIDDEN, anim_rect, border_radius=8)

                # Draw static matched card
                if card["state"] == "matched":
                    pygame.draw.rect(self.screen, self.COLOR_CARD_MATCHED, rect, border_radius=8)
                    self._draw_symbol(card["symbol"], rect, self.COLOR_GRID)
                    # Glow effect
                    glow_color = (*self.COLOR_MATCH, int(100 * (1 - card["match_glow"])))
                    glow_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
                    pygame.draw.rect(glow_surface, glow_color, glow_surface.get_rect(), border_radius=8)
                    self.screen.blit(glow_surface, rect.topleft)

        # Draw cursor
        if not self.game_over:
            cursor_r, cursor_c = self.cursor_pos
            cursor_rect = self.grid[cursor_r][cursor_c]["rect"]
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 4, border_radius=10)
        
        # Draw mismatch indicator
        if self.mismatch_cooldown > 0:
            r1, c1 = self.last_mismatched_pair[0]
            r2, c2 = self.last_mismatched_pair[1]
            rect1 = self.grid[r1][c1]["rect"]
            rect2 = self.grid[r2][c2]["rect"]
            alpha = min(255, int(512 * (self.mismatch_cooldown / (self.FPS * 0.75))))
            color = (*self.COLOR_MISMATCH, alpha)
            
            self._draw_thick_line(rect1.center, rect2.center, 5, color)


    def _draw_symbol(self, symbol, rect, color=None):
        if color is None:
            color = self.COLOR_SYMBOL
        center = rect.center
        size = rect.width * 0.35

        if symbol == 0: # Circle
            pygame.gfxdraw.aacircle(self.screen, int(center[0]), int(center[1]), int(size), color)
            pygame.gfxdraw.filled_circle(self.screen, int(center[0]), int(center[1]), int(size), color)
        elif symbol == 1: # Square
            sq_rect = pygame.Rect(0, 0, size * 2, size * 2)
            sq_rect.center = center
            pygame.draw.rect(self.screen, color, sq_rect, border_radius=int(size*0.2))
        elif symbol == 2: # Triangle
            points = [
                (center[0], center[1] - size),
                (center[0] - size, center[1] + size * 0.7),
                (center[0] + size, center[1] + size * 0.7)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif symbol == 3: # X
            self._draw_thick_line((center[0]-size, center[1]-size), (center[0]+size, center[1]+size), int(size*0.3), color)
            self._draw_thick_line((center[0]-size, center[1]+size), (center[0]+size, center[1]-size), int(size*0.3), color)
        elif symbol == 4: # Diamond
            points = [
                (center[0], center[1] - size * 1.2), (center[0] + size * 1.2, center[1]),
                (center[0], center[1] + size * 1.2), (center[0] - size * 1.2, center[1])
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif symbol == 5: # Plus
            pygame.draw.rect(self.screen, color, (center[0]-size, center[1]-size*0.15, size*2, size*0.3))
            pygame.draw.rect(self.screen, color, (center[0]-size*0.15, center[1]-size, size*0.3, size*2))
        elif symbol == 6: # Star
            self._draw_star(center, 5, size*1.4, size*0.6, 0, color)
        elif symbol == 7: # Hexagon
            points = []
            for i in range(6):
                angle_deg = 60 * i - 30
                angle_rad = math.pi / 180 * angle_deg
                points.append((center[0] + size * 1.2 * math.cos(angle_rad),
                               center[1] + size * 1.2 * math.sin(angle_rad)))
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _draw_thick_line(self, p1, p2, thickness, color):
        x1, y1 = p1
        x2, y2 = p2
        center_L = ((x1+x2)/2, (y1+y2)/2)
        length = math.dist(p1, p2)
        angle = math.atan2(y1-y2, x1-x2)
        
        UL = (center_L[0] + (length/2.) * math.cos(angle) - (thickness/2.) * math.sin(angle),
              center_L[1] + (thickness/2.) * math.cos(angle) + (length/2.) * math.sin(angle))
        UR = (center_L[0] - (length/2.) * math.cos(angle) - (thickness/2.) * math.sin(angle),
              center_L[1] + (thickness/2.) * math.cos(angle) - (length/2.) * math.sin(angle))
        BL = (center_L[0] + (length/2.) * math.cos(angle) + (thickness/2.) * math.sin(angle),
              center_L[1] - (thickness/2.) * math.cos(angle) + (length/2.) * math.sin(angle))
        BR = (center_L[0] - (length/2.) * math.cos(angle) + (thickness/2.) * math.sin(angle),
              center_L[1] - (thickness/2.) * math.cos(angle) - (length/2.) * math.sin(angle))
        
        points = [UL, UR, BR, BL]
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _draw_star(self, center, n_points, outer_radius, inner_radius, angle, color):
        points = []
        for i in range(2 * n_points):
            r = outer_radius if i % 2 == 0 else inner_radius
            curr_angle = angle + i * math.pi / n_points
            x = center[0] + int(r * math.cos(curr_angle))
            y = center[1] + int(r * math.sin(curr_angle))
            points.append((x, y))
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        
    def _render_ui(self):
        # UI Background
        pygame.draw.rect(self.screen, (0,0,0,100), (0, 0, self.WIDTH, 50))
        
        # Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 12))
        
        # Matches
        matches_text = self.font_medium.render(f"MATCHES: {self.matches_found}/8", True, self.COLOR_TEXT)
        matches_rect = matches_text.get_rect(centerx=self.WIDTH/2, top=12)
        self.screen.blit(matches_text, matches_rect)
        
        # Timer
        time_left = max(0, math.ceil(self.timer / self.FPS))
        timer_color = self.COLOR_TEXT
        if time_left <= 10: timer_color = self.COLOR_TIMER_DANGER
        elif time_left <= 20: timer_color = self.COLOR_TIMER_WARN
        
        timer_text = self.font_medium.render(f"TIME: {time_left}", True, timer_color)
        timer_rect = timer_text.get_rect(right=self.WIDTH - 15, top=12)
        self.screen.blit(timer_text, timer_rect)

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                msg = "YOU WIN!"
                color = self.COLOR_MATCH
            else:
                msg = "TIME'S UP!"
                color = self.COLOR_MISMATCH
                
            end_text = self.font_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(end_text, end_rect)

            final_score_text = self.font_medium.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            final_score_rect = final_score_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 30))
            self.screen.blit(final_score_text, final_score_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": max(0, math.ceil(self.timer / self.FPS)),
            "matches_found": self.matches_found,
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a persistent display window
    pygame.display.set_caption("Memory Match Gym Environment")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    running = True
    
    # Game loop
    while running:
        action = [0, 0, 0] # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Reset on termination after a delay
        if terminated:
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False

        env.clock.tick(env.FPS)
        
    env.close()