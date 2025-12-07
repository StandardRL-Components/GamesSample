import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:30:44.149536
# Source Brief: brief_02215.md
# Brief Index: 2215
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: Decipher evolving fractal codes by traversing temporal portals.

    The agent observes a slice of a fractal's evolution through time.
    By selecting portals, it can view the fractal at different time slices.
    The shape of the fractal at a given time is determined by a hidden code.
    The agent's goal is to observe these changes and correctly input the hidden code.

    - Movement (actions[0]): Selects one of four time portals.
    - Space (actions[1]): Inputs code character 'A'.
    - Shift (actions[2]): Inputs code character 'B'.
    """
    game_description = (
        "Decipher an evolving fractal code by traveling through temporal portals to observe its structure at different points in time."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to select a time portal. Press space to input code 'A' and shift to input code 'B'."
    )
    auto_advance = True
    metadata = {"render_modes": ["rgb_array"]}

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    INITIAL_CODE_LENGTH = 3
    MAX_CODE_LENGTH = 8
    MAX_INCORRECT_GUESSES = 3
    MAX_EPISODE_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (15, 18, 28)
    COLOR_GRID = (30, 35, 50)
    COLOR_UI_TEXT = (200, 200, 220)
    COLOR_PORTAL_NEUTRAL = (100, 100, 120)
    COLOR_PORTAL_PAST = (80, 120, 255)
    COLOR_PORTAL_FUTURE = (255, 100, 80)
    COLOR_FRACTAL_PRESENT = (0, 255, 180)
    COLOR_FRACTAL_TIME = (255, 255, 255)
    COLOR_GUESS_CORRECT = (255, 215, 0) # Gold
    COLOR_GUESS_INCORRECT = (255, 50, 50)
    COLOR_GUESS_NORMAL = (180, 180, 180)
    COLOR_GUESS_SLOT = (50, 60, 80)
    COLOR_ERROR = (200, 0, 0)
    
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
        
        # --- Fonts ---
        try:
            self.font_main = pygame.font.SysFont("Consolas", 24)
            self.font_small = pygame.font.SysFont("Consolas", 18)
            self.font_code = pygame.font.SysFont("monospace", 30)
        except pygame.error:
            self.font_main = pygame.font.SysFont(None, 28)
            self.font_small = pygame.font.SysFont(None, 22)
            self.font_code = pygame.font.SysFont(None, 34)

        # --- Persistent State (across resets) ---
        self._current_code_length = self.INITIAL_CODE_LENGTH
        
        # --- Game State (reset each episode) ---
        self.steps = 0
        self.score = 0
        self.secret_code = []
        self.player_guess = []
        self.current_time_slice = 0
        self.incorrect_final_guesses = 0
        self.portals = []
        self.last_action_feedback = ""
        self.last_action_timer = 0
        
        # --- Action state tracking ---
        self.prev_space_held = False
        self.prev_shift_held = False

        # self.reset() is called by the environment wrapper
        # self.validate_implementation() is for debugging, not needed in final version

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        
        # On failure, reset difficulty
        if options and options.get("failure", False):
            self._current_code_length = self.INITIAL_CODE_LENGTH
        # On success, increase difficulty
        elif options and options.get("success", False):
            self._current_code_length = min(self.MAX_CODE_LENGTH, self._current_code_length + 1)
        
        self.secret_code = [random.choice(['A', 'B']) for _ in range(self._current_code_length)]
        self.player_guess = []
        self.current_time_slice = 0
        self.incorrect_final_guesses = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self._generate_portals()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.steps += 1
        reward = 0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Actions ---
        action_taken = False
        
        # 1. Portal Selection
        if movement in [1, 2, 3, 4]:
            portal_index = movement - 1
            if portal_index < len(self.portals):
                self.current_time_slice = self.portals[portal_index]['target_slice']
                action_taken = True
                # SFX: whoosh or portal activation sound
        
        # 2. Code Guess (rising edge detection)
        press_A = space_held and not self.prev_space_held
        press_B = shift_held and not self.prev_shift_held
        
        guess_made = None
        if press_A and not press_B: guess_made = 'A'
        if press_B and not press_A: guess_made = 'B'
        
        if guess_made and len(self.player_guess) < self._current_code_length:
            self.player_guess.append(guess_made)
            action_taken = True
            # SFX: UI click sound
            
            # Immediate reward for correct segment
            guess_idx = len(self.player_guess) - 1
            if self.player_guess[guess_idx] == self.secret_code[guess_idx]:
                reward += 1.0
                # SFX: positive chime
            else:
                # SFX: negative buzz
                pass

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        # --- Check for Termination ---
        terminated = False
        is_success = False
        
        # Full code submitted
        if len(self.player_guess) == self._current_code_length:
            if self.player_guess == self.secret_code:
                # VICTORY
                reward += 100.0 # Large terminal reward
                self.score += self._current_code_length * 10
                terminated = True
                is_success = True
                # SFX: success fanfare
                # On the next reset, difficulty will increase
                # We need to capture the final state *before* resetting
                obs = self._get_observation(is_final_state=True, is_success=True)
                self.reset(options={"success": True})
                return obs, reward, terminated, False, self._get_info()
            else:
                # INCORRECT SUBMISSION
                self.incorrect_final_guesses += 1
                self.player_guess = [] # Reset for another try
                # SFX: loud error sound
                if self.incorrect_final_guesses >= self.MAX_INCORRECT_GUESSES:
                    # FAILURE
                    reward = -100.0 # Large terminal penalty
                    terminated = True
                    # On the next reset, difficulty will be reset
                    obs = self._get_observation(is_final_state=True, is_success=False)
                    self.reset(options={"failure": True})
                    return obs, reward, terminated, False, self._get_info()

        # Max steps termination
        truncated = self.steps >= self.MAX_EPISODE_STEPS
        if truncated:
            terminated = True # Gymnasium standard is that truncated implies terminated
            
        # If terminated due to win, we've already reset. Don't return the new state.
        # We must return the state that *led* to termination.
        # So we render *before* the potential internal reset call.
        observation = self._get_observation(is_final_state=terminated, is_success=is_success)

        return observation, reward, terminated, truncated, self._get_info()

    def _generate_portals(self):
        self.portals = []
        max_time = self._current_code_length
        
        # Generate distinct time slices to visit
        possible_slices = list(range(-max_time, max_time + 1))
        if self.current_time_slice in possible_slices:
            possible_slices.remove(self.current_time_slice)
        
        # Ensure we have options even if list is small
        if not possible_slices: possible_slices = [0]
        
        # Create 4 portals with unique targets if possible
        targets = set()
        while len(targets) < 4 and len(targets) < len(possible_slices):
            targets.add(random.choice(possible_slices))
            
        portal_targets = sorted(list(targets))
        while len(portal_targets) < 4:
            portal_targets.append(portal_targets[-1] if portal_targets else 0)

        positions = [
            (self.SCREEN_WIDTH * 0.2, self.SCREEN_HEIGHT * 0.5), # Left
            (self.SCREEN_WIDTH * 0.8, self.SCREEN_HEIGHT * 0.5), # Right
            (self.SCREEN_WIDTH * 0.5, self.SCREEN_HEIGHT * 0.2), # Top
            (self.SCREEN_WIDTH * 0.5, self.SCREEN_HEIGHT * 0.8)  # Bottom
        ]
        
        # Map movement actions to portals
        # 1=Up, 2=Down, 3=Left, 4=Right
        portal_map = { 1: 2, 2: 3, 3: 0, 4: 1 }
        
        for i in range(4):
            self.portals.append({
                'pos': positions[portal_map[i+1]],
                'target_slice': portal_targets[i]
            })

    def _get_observation(self, is_final_state=False, is_success=False):
        self.screen.fill(self.COLOR_BG)
        self._draw_grid()
        
        self._draw_portals()
        self._draw_fractal()
        
        self._render_ui()
        
        if is_final_state:
            self._draw_end_screen(is_success)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "code_length": self._current_code_length}

    def _draw_grid(self):
        for i in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))

    def _draw_portals(self):
        for i, portal in enumerate(self.portals):
            pos = (int(portal['pos'][0]), int(portal['pos'][1]))
            target = portal['target_slice']
            
            # Color based on time direction
            if target < self.current_time_slice: color = self.COLOR_PORTAL_PAST
            elif target > self.current_time_slice: color = self.COLOR_PORTAL_FUTURE
            else: color = self.COLOR_PORTAL_NEUTRAL
            
            # Shimmer effect
            radius_anim = 5 * math.sin(self.steps * 0.1 + i)
            base_radius = 25
            
            # Draw shimmering rings
            for j in range(3):
                r = base_radius + radius_anim * (j * 0.5)
                alpha = 100 - j * 30
                c = (*color, alpha)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(max(0, r + j*5)), c)

            # Draw central circle
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(base_radius), self.COLOR_BG)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(base_radius), color)
            
            # Draw portal target time
            text = self.font_small.render(str(target), True, self.COLOR_UI_TEXT)
            text_rect = text.get_rect(center=pos)
            self.screen.blit(text, text_rect)

    def _draw_fractal(self):
        start_pos = (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self._fractal_recursive(start_pos, -90, 60, 0)
        
    def _fractal_recursive(self, pos, angle, length, depth):
        if depth >= self._current_code_length or length < 2:
            return

        # Fractal shape is only revealed up to the absolute time slice
        is_revealed = depth < abs(self.current_time_slice)
        code_char = self.secret_code[depth] if is_revealed else 'N' # Neutral
        
        # Time affects the angles
        time_factor = 2.0
        time_influence = self.current_time_slice * time_factor

        if code_char == 'A':
            angle_mod1, angle_mod2 = 25, -25
        elif code_char == 'B':
            angle_mod1, angle_mod2 = -25, 25
        else: # Neutral / unrevealed
            angle_mod1, angle_mod2 = 15, -15

        # Calculate branches
        branches = [
            (angle + angle_mod1 + time_influence, length * 0.75),
            (angle + angle_mod2 - time_influence, length * 0.75)
        ]

        # Determine color based on depth and time
        lerp_factor = min(1, abs(self.current_time_slice) / max(1, self._current_code_length))
        color = tuple(int(c1 * (1-lerp_factor) + c2 * lerp_factor) for c1, c2 in zip(self.COLOR_FRACTAL_PRESENT, self.COLOR_FRACTAL_TIME))

        for ang, L in branches:
            rad = math.radians(ang)
            end_pos = (pos[0] + L * math.cos(rad), pos[1] + L * math.sin(rad))
            pygame.draw.aaline(self.screen, color, pos, end_pos, 2)
            self._fractal_recursive(end_pos, ang, L, depth + 1)

    def _render_ui(self):
        # --- Score ---
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - 160, 10))

        # --- Time Slice ---
        time_text = self.font_main.render(f"Time Slice: {self.current_time_slice}", True, self.COLOR_UI_TEXT)
        time_rect = time_text.get_rect(center=(self.SCREEN_WIDTH / 2, 25))
        self.screen.blit(time_text, time_rect)
        
        # --- Incorrect Guesses ---
        errors_text = self.font_small.render("Attempts:", True, self.COLOR_UI_TEXT)
        self.screen.blit(errors_text, (10, 10))
        for i in range(self.MAX_INCORRECT_GUESSES):
            color = self.COLOR_ERROR if i < self.incorrect_final_guesses else self.COLOR_GUESS_SLOT
            pygame.draw.circle(self.screen, color, (100 + i * 20, 20), 7)

        # --- Code Input Display ---
        total_width = self._current_code_length * 40
        start_x = (self.SCREEN_WIDTH - total_width) / 2
        
        for i in range(self._current_code_length):
            x = start_x + i * 40
            y = self.SCREEN_HEIGHT - 50
            slot_rect = pygame.Rect(x, y, 35, 45)
            
            if i < len(self.player_guess):
                char = self.player_guess[i]
                is_correct = char == self.secret_code[i]
                color = self.COLOR_GUESS_CORRECT if is_correct else self.COLOR_GUESS_NORMAL
                
                text = self.font_code.render(char, True, color)
                text_rect = text.get_rect(center=slot_rect.center)
                pygame.draw.rect(self.screen, self.COLOR_GUESS_SLOT, slot_rect, border_radius=5)
                self.screen.blit(text, text_rect)
            else:
                pygame.draw.rect(self.screen, self.COLOR_GUESS_SLOT, slot_rect, border_radius=5)
                text = self.font_code.render("_", True, self.COLOR_GUESS_NORMAL)
                text_rect = text.get_rect(center=slot_rect.center)
                self.screen.blit(text, text_rect)

    def _draw_end_screen(self, is_success):
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        s.fill((0,0,0, 180))
        
        message = "CODE DECIPHERED" if is_success else "SYSTEM LOCKOUT"
        color = self.COLOR_GUESS_CORRECT if is_success else self.COLOR_ERROR
        
        end_text = self.font_main.render(message, True, color)
        end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
        self.screen.blit(s, (0,0))
        self.screen.blit(end_text, end_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Manual Play Example ---
    # This block is not part of the Gymnasium environment but is useful for testing.
    # To run, you need to unset the dummy video driver.
    # For example, run the script with: SDL_VIDEODRIVER=x11 python your_script_name.py
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Fractal Code Decipherer")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    print("\n--- Controls ---")
    print("Arrows: Select Portals")
    print("Space: Input 'A'")
    print("Shift: Input 'B'")
    print("R: Reset Environment")
    print("----------------\n")
    
    while not terminated:
        # --- Action Mapping for Manual Play ---
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Environment Reset ---")
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        # --- Step Environment ---
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")

        if term:
            print(f"--- Episode Finished ---")
            print(f"Final Score: {info['score']}, Final Total Reward: {total_reward:.2f}")
            # In manual play, we might want to pause before the next episode
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        # --- Render ---
        # The observation is already the rendered image
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS for manual play
        
    env.close()