
# Generated: 2025-08-28T02:01:38.078839
# Source Brief: brief_01575.md
# Brief Index: 1575

        
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
        "A fast-paced memory game. Race against the clock to find all matching pairs of geometric patterns."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.render_mode = render_mode
        
        # Game constants
        self.FPS = 30
        self.MAX_TIME_SECONDS = 60
        self.MAX_STEPS = self.MAX_TIME_SECONDS * self.FPS
        
        self.GRID_ROWS, self.GRID_COLS = 4, 4
        self.NUM_PAIRS = (self.GRID_ROWS * self.GRID_COLS) // 2
        
        self.CARD_SIZE = 70
        self.CARD_GAP = 15
        
        grid_width = self.GRID_COLS * self.CARD_SIZE + (self.GRID_COLS - 1) * self.CARD_GAP
        grid_height = self.GRID_ROWS * self.CARD_SIZE + (self.GRID_ROWS - 1) * self.CARD_GAP
        self.GRID_TOP_LEFT_X = (self.SCREEN_WIDTH - grid_width) // 2
        self.GRID_TOP_LEFT_Y = (self.SCREEN_HEIGHT - grid_height) // 2 + 20

        # Colors
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_CARD_BACK = (70, 80, 100)
        self.COLOR_CARD_FACE = (40, 45, 55)
        self.COLOR_CURSOR = (255, 204, 0)
        self.COLOR_MATCH_GLOW = (60, 255, 160)
        self.COLOR_TEXT = (220, 220, 220)
        self.PATTERN_COLORS = [
            (255, 87, 87), (255, 165, 0), (255, 255, 0), (138, 255, 87),
            (87, 187, 255), (87, 87, 255), (187, 87, 255), (255, 87, 187)
        ]
        self.SHAPES = ['circle', 'square', 'triangle', 'diamond']

        # Fonts
        self.UI_FONT = pygame.font.SysFont("Consolas", 24, bold=True)
        self.MSG_FONT = pygame.font.SysFont("Arial", 48, bold=True)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        self.cursor_pos = [0, 0]
        self.cards = []
        self.selected_cards_indices = []
        self.matched_pairs = 0
        self.last_space_held = False
        self.cursor_move_cooldown = 0
        self.mismatch_check_timer = 0
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_TIME_SECONDS * self.FPS
        self.cursor_pos = [0, 0]
        self.selected_cards_indices = []
        self.matched_pairs = 0
        self.last_space_held = False
        self.cursor_move_cooldown = 0
        self.mismatch_check_timer = 0
        
        self._create_cards()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.01  # Small penalty for each step to encourage speed

        self.steps += 1
        if not self.game_over:
            self.time_remaining -= 1

        # Update card flip animations
        for card in self.cards:
            if card['flip_progress'] != card['target_flip']:
                card['flip_progress'] += (card['target_flip'] - card['flip_progress']) * 0.25
                # Snap to target if very close
                if abs(card['flip_progress'] - card['target_flip']) < 0.01:
                    card['flip_progress'] = card['target_flip']

        # Handle mismatch check delay
        if self.mismatch_check_timer > 0:
            self.mismatch_check_timer -= 1
            if self.mismatch_check_timer == 0:
                card1_idx, card2_idx = self.selected_cards_indices
                card1, card2 = self.cards[card1_idx], self.cards[card2_idx]
                
                if card1['pattern_id'] == card2['pattern_id']:
                    # MATCH
                    card1['state'] = 'matched'
                    card2['state'] = 'matched'
                    self.matched_pairs += 1
                    reward += 5.0
                    self.score += 5
                    # Sound: Match success
                else:
                    # MISMATCH
                    card1['target_flip'] = 0.0
                    card2['target_flip'] = 0.0
                    card1['state'] = 'down'
                    card2['state'] = 'down'
                    reward -= 1.0
                    self.score -= 1
                    # Sound: Mismatch fail
                
                self.selected_cards_indices.clear()

        # Handle player input only if not checking a mismatch
        if self.mismatch_check_timer == 0 and not self.game_over:
            # Cursor movement
            if self.cursor_move_cooldown > 0:
                self.cursor_move_cooldown -= 1
            
            if movement != 0 and self.cursor_move_cooldown == 0:
                if movement == 1: self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_ROWS) % self.GRID_ROWS
                elif movement == 2: self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_ROWS
                elif movement == 3: self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_COLS) % self.GRID_COLS
                elif movement == 4: self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_COLS
                self.cursor_move_cooldown = 4 # frames

            # Card selection
            space_pressed = space_held and not self.last_space_held
            if space_pressed and len(self.selected_cards_indices) < 2:
                card_idx = self.cursor_pos[0] * self.GRID_COLS + self.cursor_pos[1]
                card = self.cards[card_idx]
                if card['state'] == 'down':
                    card['state'] = 'up'
                    card['target_flip'] = 1.0
                    self.selected_cards_indices.append(card_idx)
                    # Sound: Flip card
                    
                    if len(self.selected_cards_indices) == 2:
                        self.mismatch_check_timer = self.FPS // 2 # 0.5 sec delay
        
        self.last_space_held = space_held
        
        # Check termination conditions
        terminated = False
        if self.matched_pairs == self.NUM_PAIRS:
            terminated = True
            reward += 50.0
            self.score += 50
        elif self.time_remaining <= 0:
            terminated = True
            reward -= 50.0
            self.score -= 50
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        if terminated:
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _create_cards(self):
        self.cards.clear()
        
        # Define 8 unique patterns for this game instance
        all_patterns = [(c, s) for c in range(len(self.PATTERN_COLORS)) for s in range(len(self.SHAPES))]
        self.np_random.shuffle(all_patterns)
        game_patterns = all_patterns[:self.NUM_PAIRS]
        
        # Create card pairs
        card_pattern_ids = list(range(self.NUM_PAIRS)) * 2
        self.np_random.shuffle(card_pattern_ids)
        
        for i, pattern_id in enumerate(card_pattern_ids):
            row, col = divmod(i, self.GRID_COLS)
            x = self.GRID_TOP_LEFT_X + col * (self.CARD_SIZE + self.CARD_GAP)
            y = self.GRID_TOP_LEFT_Y + row * (self.CARD_SIZE + self.CARD_GAP)
            
            self.cards.append({
                'rect': pygame.Rect(x, y, self.CARD_SIZE, self.CARD_SIZE),
                'pattern_id': pattern_id,
                'game_pattern': game_patterns[pattern_id],
                'state': 'down',
                'flip_progress': 0.0, # 0.0 is face down, 1.0 is face up
                'target_flip': 0.0,
            })
            
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
            "time_remaining": self.time_remaining / self.FPS,
            "matched_pairs": self.matched_pairs,
        }

    def _render_game(self):
        # Draw cards
        for i, card in enumerate(self.cards):
            self._draw_card(card)

        # Draw cursor
        if not self.game_over:
            cursor_idx = self.cursor_pos[0] * self.GRID_COLS + self.cursor_pos[1]
            cursor_rect = self.cards[cursor_idx]['rect']
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 4, border_radius=8)

    def _draw_card(self, card):
        rect = card['rect']
        
        # Matched cards are faded with a glow
        if card['state'] == 'matched':
            glow_rect = rect.inflate(8, 8)
            pygame.draw.rect(self.screen, self.COLOR_MATCH_GLOW, glow_rect, 0, border_radius=12)
            pygame.draw.rect(self.screen, self.COLOR_CARD_FACE, rect, 0, border_radius=8)
            self._draw_pattern(card, rect, 100) # Draw pattern with alpha
            return

        # Flip animation
        angle = card['flip_progress'] * math.pi
        width_scale = math.cos(angle)
        
        if width_scale == 0: return # Card is edge-on

        anim_rect = rect.copy()
        anim_rect.width = int(rect.width * abs(width_scale))
        anim_rect.centerx = rect.centerx

        # Draw back or face based on which way it's flipped
        if width_scale > 0: # Face down
            pygame.draw.rect(self.screen, self.COLOR_CARD_BACK, anim_rect, 0, border_radius=8)
        else: # Face up
            pygame.draw.rect(self.screen, self.COLOR_CARD_FACE, anim_rect, 0, border_radius=8)
            # Only draw pattern if the card is wide enough to see
            if anim_rect.width > 10:
                self._draw_pattern(card, anim_rect)

    def _draw_pattern(self, card, rect, alpha=255):
        color_idx, shape_idx = card['game_pattern']
        color = self.PATTERN_COLORS[color_idx]
        shape = self.SHAPES[shape_idx]
        
        # Apply alpha
        if alpha < 255:
            color = (*color, alpha)

        center = rect.center
        size = int(rect.width * 0.6)

        if shape == 'circle':
            pygame.gfxdraw.aacircle(self.screen, center[0], center[1], size // 2, color)
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], size // 2, color)
        elif shape == 'square':
            sq_rect = pygame.Rect(0, 0, size, size)
            sq_rect.center = center
            # gfxdraw doesn't have a filled rect with alpha, so we use a surface
            s = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.rect(s, color, s.get_rect())
            self.screen.blit(s, sq_rect.topleft)
        elif shape == 'triangle':
            points = [
                (center[0], center[1] - size // 2),
                (center[0] - size // 2, center[1] + size // 2),
                (center[0] + size // 2, center[1] + size // 2),
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif shape == 'diamond':
            points = [
                (center[0], center[1] - size // 2),
                (center[0] + size // 2, center[1]),
                (center[0], center[1] + size // 2),
                (center[0] - size // 2, center[1]),
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_ui(self):
        # Draw score/matched pairs
        pairs_text = f"Pairs: {self.matched_pairs}/{self.NUM_PAIRS}"
        pairs_surf = self.UI_FONT.render(pairs_text, True, self.COLOR_TEXT)
        self.screen.blit(pairs_surf, (20, 15))

        # Draw timer
        secs = max(0, int(self.time_remaining / self.FPS))
        mins = secs // 60
        secs %= 60
        timer_text = f"Time: {mins:02}:{secs:02}"
        timer_surf = self.UI_FONT.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (self.SCREEN_WIDTH - timer_surf.get_width() - 20, 15))

        # Draw game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            if self.matched_pairs == self.NUM_PAIRS:
                msg_text = "YOU WIN!"
                msg_color = self.COLOR_MATCH_GLOW
            else:
                msg_text = "TIME'S UP!"
                msg_color = self.PATTERN_COLORS[0]
            
            msg_surf = self.MSG_FONT.render(msg_text, True, msg_color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Memory Match")
    clock = pygame.time.Clock()
    
    print(env.user_guide)
    print(env.game_description)

    action = env.action_space.sample()
    action[0] = 0 # No movement
    action[1] = 0 # Space not held
    action[2] = 0 # Shift not held

    while not done:
        # Human input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        
        # Reset actions
        action[0] = 0 # Movement
        action[1] = 0 # Space
        
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    env.close()