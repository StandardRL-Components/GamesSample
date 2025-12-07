import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:33:47.249097
# Source Brief: brief_01812.md
# Brief Index: 1812
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A dice-rolling board game where you build and spend momentum to reach the end of the track."
    )
    user_guide = (
        "Use arrow keys to choose your base roll (e.g., ↑ for high, ↓ for low). "
        "Press space to add +1 to the final roll."
    )
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.FPS = 30
        self.NUM_SPACES = 50
        self.MAX_ROLLS = 15
        self.WIN_SCORE_BONUS = 25
        self.WIN_REWARD = 100
        self.FALL_OFF_REWARD = -100
        self.OUT_OF_ROLLS_REWARD = -50
        self.FORWARD_MOVE_REWARD_FACTOR = 0.1

        # --- Colors ---
        self.COLOR_BG = (44, 62, 80) # Dark Blue/Gray
        self.COLOR_BOARD = (127, 140, 141) # Mid Gray
        self.COLOR_PAWN = (52, 152, 219) # Bright Blue
        self.COLOR_BONUS = (46, 204, 113) # Green
        self.COLOR_PENALTY = (231, 76, 60) # Red
        self.COLOR_TEXT = (236, 240, 241) # White
        self.COLOR_UI_BG = (0, 0, 0, 128)
        self.COLOR_MOMENTUM_POS = (39, 174, 96) # Green for momentum
        self.COLOR_MOMENTUM_NEG = (192, 57, 43) # Red for momentum

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("Consolas", 16)
        self.font_m = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- Board Rendering ---
        self.SPACE_WIDTH = 80
        self.SPACE_HEIGHT = 100
        self.BOARD_Y_POS = self.SCREEN_HEIGHT // 2
        self.board_surface = pygame.Surface((self.NUM_SPACES * self.SPACE_WIDTH, self.SPACE_HEIGHT))

        # --- Player Rendering ---
        self.PAWN_SCREEN_X = self.SCREEN_WIDTH // 4
        
        # --- State Variables ---
        self.pawn_position = 0
        self.pawn_visual_pos = 0.0
        self.momentum = 0
        self.score = 0
        self.rolls_left = 0
        self.last_dice_roll = 0
        self.game_over = False
        self.win_condition = False
        self.board_spaces = []
        self.dice_animation_timer = 0
        self.last_reward_text = ""
        self.last_reward_timer = 0
        
        # --- Initialize State ---
        # self.reset() is called by the environment wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.pawn_position = 0
        self.pawn_visual_pos = 0.0
        self.momentum = 0
        self.score = 0
        self.rolls_left = self.MAX_ROLLS
        self.last_dice_roll = 0
        self.game_over = False
        self.win_condition = False
        self.dice_animation_timer = 0
        self.last_reward_text = ""
        self.last_reward_timer = 0

        self._generate_board()
        self._pre_render_board()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Unpack and interpret action
        movement, space_held, _ = action
        base_roll = movement + 1
        dice_roll = min(base_roll + (1 if space_held else 0), 6)
        self.last_dice_roll = dice_roll
        self.dice_animation_timer = self.FPS # Animate for 1 second

        # 2. Update game state
        self.rolls_left -= 1
        
        move_amount = dice_roll + self.momentum
        old_position = self.pawn_position
        self.pawn_position += move_amount

        # Update momentum: decays by 1, modified by roll
        self.momentum += (dice_roll - 3) # Rolls > 3 increase, < 3 decrease
        self.momentum -= 1 # Natural decay

        # 3. Calculate reward and check for termination
        reward = 0
        terminated = False
        
        # Reward for movement
        reward += move_amount * self.FORWARD_MOVE_REWARD_FACTOR

        # Check termination conditions
        if self.pawn_position >= self.NUM_SPACES - 1:
            self.pawn_position = self.NUM_SPACES - 1
            reward += self.WIN_REWARD
            self.score += self.WIN_SCORE_BONUS
            self.game_over = True
            self.win_condition = True
            self._show_reward_text(f"+{self.WIN_REWARD} WIN!", self.COLOR_BONUS)
        elif self.pawn_position < 0:
            reward += self.FALL_OFF_REWARD
            self.game_over = True
            self._show_reward_text(f"{self.FALL_OFF_REWARD} FALL!", self.COLOR_PENALTY)
        elif self.rolls_left <= 0:
            reward += self.OUT_OF_ROLLS_REWARD
            self.game_over = True
            self._show_reward_text("OUT OF ROLLS", self.COLOR_PENALTY)
        else: # Not terminated, check for space bonus/penalty
            space_type = self.board_spaces[self.pawn_position]
            if space_type == 'bonus':
                reward += 5
                self.score += 5
                self._show_reward_text("+5 BONUS", self.COLOR_BONUS)
                # Sound: bonus_pickup.wav
            elif space_type == 'penalty':
                penalty_val = min(0, -2 + self.momentum) # Penalty reduced by momentum
                reward += penalty_val
                self.score += penalty_val
                self._show_reward_text(f"{int(penalty_val)} PENALTY", self.COLOR_PENALTY)
                # Sound: penalty_hit.wav

        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        # Update timers for animations
        if self.dice_animation_timer > 0: self.dice_animation_timer -= 1
        if self.last_reward_timer > 0: self.last_reward_timer -= 1

        # Smooth pawn visual movement (interpolation)
        lerp_factor = 0.15
        self.pawn_visual_pos = (self.pawn_visual_pos * (1 - lerp_factor) + 
                                self.pawn_position * lerp_factor)

        # --- Rendering ---
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "rolls_left": self.rolls_left,
            "momentum": self.momentum,
            "pawn_position": self.pawn_position,
        }

    def _generate_board(self):
        self.board_spaces = ['normal'] * self.NUM_SPACES
        # Ensure start and end are normal
        self.board_spaces[0] = 'start'
        self.board_spaces[-1] = 'end'
        
        possible_indices = list(range(1, self.NUM_SPACES - 1))
        # Use self.np_random for reproducibility if a seed is set
        if self.np_random:
            self.np_random.shuffle(possible_indices)
        else:
            random.shuffle(possible_indices)

        
        num_bonus = 6
        num_penalty = 8
        
        for i in range(num_bonus):
            self.board_spaces[possible_indices.pop()] = 'bonus'
        for i in range(num_penalty):
            self.board_spaces[possible_indices.pop()] = 'penalty'

    def _pre_render_board(self):
        self.board_surface.fill(self.COLOR_BG)
        for i in range(self.NUM_SPACES):
            space_type = self.board_spaces[i]
            rect = pygame.Rect(i * self.SPACE_WIDTH, 0, self.SPACE_WIDTH, self.SPACE_HEIGHT)
            
            color = self.COLOR_BOARD
            if space_type == 'bonus': color = self.COLOR_BONUS
            elif space_type == 'penalty': color = self.COLOR_PENALTY
            
            # Draw main space with rounded corners
            pygame.draw.rect(self.board_surface, color, rect.inflate(-8, -8), border_radius=10)
            
            # Draw number
            num_text = self.font_m.render(str(i + 1), True, self.COLOR_TEXT)
            text_rect = num_text.get_rect(center=rect.center)
            self.board_surface.blit(num_text, text_rect)

    def _render_game(self):
        # Calculate camera offset to keep pawn centered
        camera_x = self.pawn_visual_pos * self.SPACE_WIDTH
        board_render_pos = (self.PAWN_SCREEN_X - camera_x, self.BOARD_Y_POS - self.SPACE_HEIGHT / 2)
        
        # Render the board
        self.screen.blit(self.board_surface, board_render_pos)

        # Render the pawn with a glow effect
        pawn_center = (int(self.PAWN_SCREEN_X), int(self.BOARD_Y_POS))
        glow_radius = 25
        pawn_radius = 15
        
        # Glow effect
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_PAWN, 50), (glow_radius, glow_radius), glow_radius)
        pygame.draw.circle(glow_surf, (*self.COLOR_PAWN, 80), (glow_radius, glow_radius), int(glow_radius * 0.8))
        self.screen.blit(glow_surf, (pawn_center[0] - glow_radius, pawn_center[1] - glow_radius))
        
        # Pawn itself
        pygame.draw.circle(self.screen, self.COLOR_PAWN, pawn_center, pawn_radius)
        pygame.draw.circle(self.screen, (255,255,255), pawn_center, pawn_radius, 2)

        # Render dice roll animation
        if self.dice_animation_timer > 0:
            progress = self.dice_animation_timer / self.FPS
            alpha = int(255 * math.sin(math.pi * progress)) # Fade in and out
            size = int(80 + 40 * (1 - progress))
            self._draw_dice(self.screen, self.last_dice_roll, 
                           (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 100), 
                           size, alpha)
        
        # Render floating reward text
        if self.last_reward_timer > 0:
            progress = self.last_reward_timer / (self.FPS * 1.5)
            alpha = int(255 * progress)
            y_offset = int(50 * (1 - progress))
            
            reward_text_surf = self.font_m.render(self.last_reward_text, True, self.last_reward_color)
            reward_text_surf.set_alpha(alpha)
            text_rect = reward_text_surf.get_rect(center=(self.PAWN_SCREEN_X, self.BOARD_Y_POS - 70 - y_offset))
            self.screen.blit(reward_text_surf, text_rect)

    def _render_ui(self):
        # Draw top UI panel
        ui_panel = pygame.Surface((self.SCREEN_WIDTH, 80), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))

        # Score
        self._draw_text(f"SCORE: {self.score}", (20, 15), self.font_m, self.COLOR_TEXT)
        # Rolls Left
        self._draw_text(f"ROLLS: {self.rolls_left}/{self.MAX_ROLLS}", (20, 45), self.font_m, self.COLOR_TEXT)
        
        # Last Roll
        self._draw_text(f"LAST ROLL: {self.last_dice_roll}", (240, 15), self.font_m, self.COLOR_TEXT)
        
        # Momentum
        self._draw_text("MOMENTUM:", (450, 15), self.font_s, self.COLOR_TEXT)
        mom_bar_rect = pygame.Rect(450, 40, 170, 20)
        pygame.draw.rect(self.screen, (0,0,0), mom_bar_rect) # BG for bar
        
        max_mom = 15 # For visualization scaling
        mom_ratio = min(1, max(-1, self.momentum / max_mom))
        if mom_ratio > 0:
            bar_w = int((mom_bar_rect.width / 2) * mom_ratio)
            pygame.draw.rect(self.screen, self.COLOR_MOMENTUM_POS, (mom_bar_rect.centerx, mom_bar_rect.y, bar_w, mom_bar_rect.h))
        elif mom_ratio < 0:
            bar_w = int((mom_bar_rect.width / 2) * -mom_ratio)
            pygame.draw.rect(self.screen, self.COLOR_MOMENTUM_NEG, (mom_bar_rect.centerx - bar_w, mom_bar_rect.y, bar_w, mom_bar_rect.h))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, mom_bar_rect, 1) # Border
        
        mom_text = self.font_m.render(str(self.momentum), True, self.COLOR_TEXT)
        mom_text_rect = mom_text.get_rect(center=mom_bar_rect.center)
        self.screen.blit(mom_text, mom_text_rect)

        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = "YOU WIN!" if self.win_condition else "GAME OVER"
            color = self.COLOR_BONUS if self.win_condition else self.COLOR_PENALTY
            self._draw_text(end_text, self.screen.get_rect().center, self.font_l, color, centered=True)

    def _draw_text(self, text, pos, font, color, centered=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if centered:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _draw_dice(self, surface, value, center_pos, size, alpha):
        dice_surf = pygame.Surface((size, size), pygame.SRCALPHA)
        dice_rect = pygame.Rect(0, 0, size, size)
        
        # Background
        pygame.draw.rect(dice_surf, (*self.COLOR_TEXT, alpha), dice_rect, border_radius=int(size * 0.2))
        
        # Dots
        dot_radius = int(size * 0.1)
        dot_color = (*self.COLOR_BG, alpha)
        positions = {
            1: [(0.5, 0.5)],
            2: [(0.25, 0.25), (0.75, 0.75)],
            3: [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)],
            4: [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)],
            5: [(0.25, 0.25), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.75, 0.75)],
            6: [(0.25, 0.25), (0.75, 0.25), (0.25, 0.5), (0.75, 0.5), (0.25, 0.75), (0.75, 0.75)],
        }
        if value in positions:
            for pos in positions[value]:
                pygame.draw.circle(dice_surf, dot_color, (int(pos[0]*size), int(pos[1]*size)), dot_radius)
        
        final_rect = dice_surf.get_rect(center=center_pos)
        surface.blit(dice_surf, final_rect)

    def _show_reward_text(self, text, color):
        self.last_reward_text = text
        self.last_reward_color = color
        self.last_reward_timer = int(self.FPS * 1.5)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    # Create a display for manual play
    pygame_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Momentum Board Game")
    clock = pygame.time.Clock()

    action = [0, 0, 0] # No-op
    
    print("\n--- Manual Control ---")
    print("1-5 Keys: Select dice roll (1-5)")
    print("SPACE: Add +1 to selected roll (e.g., 5+SPACE = 6)")
    print("RETURN: Confirm roll and take turn")
    print("----------------------\n")
    
    base_roll_choice = 1

    while running:
        # --- Pygame event handling for manual play ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                
                # Map keys to actions
                if pygame.K_1 <= event.key <= pygame.K_5:
                    base_roll_choice = event.key - pygame.K_0
                
                is_space_held = pygame.key.get_pressed()[pygame.K_SPACE]

                if event.key == pygame.K_RETURN: # On Enter, take a step
                    if done:
                        obs, info = env.reset()
                        done = False
                        action = [0, 0, 0]
                        base_roll_choice = 1
                    else:
                        # Convert player choice to action tuple
                        action_mov = base_roll_choice - 1
                        action_space = 1 if is_space_held else 0
                        action = [action_mov, action_space, 0]

                        obs, reward, done, _, info = env.step(action)
                        print(
                            f"Roll: {env.last_dice_roll}, "
                            f"Pos: {info['pawn_position']}, "
                            f"Momentum: {info['momentum']}, "
                            f"Reward: {reward:.2f}, "
                            f"Score: {info['score']}"
                        )
                        if done:
                            print("\n--- GAME OVER ---")
                            print(f"Final Score: {info['score']}")
                            print("Press ENTER to play again, Q to quit.")
        
        # --- Rendering ---
        # The observation is already a rendered frame
        frame = env._get_observation()
        # Transpose back for pygame display
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        pygame_screen.blit(surf, (0, 0))
        
        # Add text for manual play instructions
        is_space_held = pygame.key.get_pressed()[pygame.K_SPACE]
        current_roll_preview = min(base_roll_choice + (1 if is_space_held else 0), 6)
        
        if not done:
            help_text_surf = env.font_m.render(f"Select Roll: {base_roll_choice} {'+ SPACE' if is_space_held else ''} = {current_roll_preview}. Press ENTER.", True, (255, 255, 0))
            help_rect = help_text_surf.get_rect(center=(env.SCREEN_WIDTH/2, env.SCREEN_HEIGHT - 30))
            pygame_screen.blit(help_text_surf, help_rect)

        pygame.display.flip()
        clock.tick(env.FPS)

    env.close()