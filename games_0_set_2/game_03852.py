
# Generated: 2025-08-28T00:37:59.972482
# Source Brief: brief_03852.md
# Brief Index: 3852

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrows to move cursor. Hold Shift over a word to grab. Press Space over a definition to release."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced word puzzle. Drag falling words to their matching definitions before they hit the bottom."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    CURSOR_SPEED = 8
    MAX_STEPS = 3600  # 2 minutes at 30fps
    WIN_MATCHES = 20
    LOSE_MISSES = 5
    NUM_DEFINITIONS = 4

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_WORD = (255, 255, 255)
    COLOR_DEF_BOX = (40, 60, 80)
    COLOR_DEF_TEXT = (200, 210, 220)
    COLOR_CURSOR = (255, 200, 0)
    COLOR_CORRECT = (0, 255, 150)
    COLOR_INCORRECT = (255, 80, 80)
    COLOR_UI_TEXT = (230, 230, 230)
    COLOR_HELD_GLOW = (255, 220, 100, 50) # RGBA

    WORD_BANK = [
        ("Agile", "Able to move quickly"),
        ("Brave", "Ready to face danger"),
        ("Calm", "Peaceful and untroubled"),
        ("Daring", "Adventurous and bold"),
        ("Eager", "Wanting to do something"),
        ("Fierce", "Savagely aggressive"),
        ("Gentle", "Kind and mild-mannered"),
        ("Happy", "Feeling pleasure"),
        ("Jovial", "Cheerful and friendly"),
        ("Keen", "Highly developed"),
        ("Lively", "Full of energy"),
        ("Merry", "Full of cheerfulness"),
        ("Noble", "Having fine qualities"),
        ("Proud", "Feeling deep satisfaction"),
        ("Quiet", "Making little noise"),
        ("Swift", "Happening quickly"),
        ("True", "In accordance with fact"),
        ("Vivid", "Producing clear images"),
        ("Wise", "Having good judgement"),
        ("Zesty", "Full of flavor"),
        ("Crisp", "Firm, dry, and brittle"),
        ("Glow", "Emit a steady light"),
        ("Huge", "Extremely large"),
        ("Icy", "Covered in ice"),
        ("Jolt", "A sudden rough push"),
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        # Fonts
        try:
            self.font_word = pygame.font.Font(None, 28)
            self.font_def = pygame.font.Font(None, 20)
            self.font_ui = pygame.font.Font(None, 24)
            self.font_game_over = pygame.font.Font(None, 64)
        except pygame.error:
            # Fallback if default font is not found (e.g., in minimal containers)
            self.font_word = pygame.font.SysFont("sans-serif", 26)
            self.font_def = pygame.font.SysFont("sans-serif", 18)
            self.font_ui = pygame.font.SysFont("sans-serif", 22)
            self.font_game_over = pygame.font.SysFont("sans-serif", 60)

        # Initialize state variables
        self.cursor_pos = None
        self.words_on_screen = None
        self.definitions_on_screen = None
        self.particles = None
        self.held_word_idx = None
        self.available_words = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.missed_words = None
        self.matched_words_count = None
        self.word_fall_speed = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.dist_to_target_on_grab = None
        self.np_random = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.missed_words = 0
        self.matched_words_count = 0
        self.word_fall_speed = 1.0

        self.cursor_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2 - 50], dtype=float)
        self.held_word_idx = -1
        self.prev_space_held = False
        self.prev_shift_held = False
        self.dist_to_target_on_grab = float('inf')

        self.words_on_screen = []
        self.definitions_on_screen = []
        self.particles = []
        
        self.available_words = self.WORD_BANK[:]
        self.np_random.shuffle(self.available_words)

        for _ in range(self.NUM_DEFINITIONS):
            self._spawn_new_word_definition_pair()
        
        return self._get_observation(), self._get_info()

    def _spawn_new_word_definition_pair(self):
        if not self.available_words:
            self.available_words = self.WORD_BANK[:]
            self.np_random.shuffle(self.available_words)

        word_text, def_text = self.available_words.pop()
        
        # Find an empty definition slot
        occupied_indices = {d['slot_idx'] for d in self.definitions_on_screen}
        available_indices = set(range(self.NUM_DEFINITIONS)) - occupied_indices
        if not available_indices:
            return # No space left, should not happen in normal flow

        slot_idx = self.np_random.choice(list(available_indices))
        
        # Create Definition
        def_width = self.WIDTH / self.NUM_DEFINITIONS - 10
        def_x = 5 + slot_idx * (def_width + 10)
        def_y = self.HEIGHT - 55
        def_rect = pygame.Rect(def_x, def_y, def_width, 50)
        self.definitions_on_screen.append({
            'text': def_text,
            'word': word_text,
            'rect': def_rect,
            'slot_idx': slot_idx
        })

        # Create Word
        word_surf = self.font_word.render(word_text, True, self.COLOR_WORD)
        word_x = self.np_random.integers(word_surf.get_width() // 2, self.WIDTH - word_surf.get_width() // 2)
        word_y = -self.np_random.integers(20, 100)
        
        self.words_on_screen.append({
            'text': word_text,
            'def_text': def_text,
            'pos': np.array([word_x, word_y], dtype=float),
            'surf': word_surf,
            'rect': word_surf.get_rect(center=(word_x, word_y))
        })

    def _create_particles(self, pos, color, count):
        # Play sound placeholder
        # if color == self.COLOR_CORRECT: print("# Play success sound")
        # elif color == self.COLOR_INCORRECT: print("# Play failure sound")
        # else: print("# Play miss sound")
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'color': color
            })

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)
            
        reward = 0
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement, space_action, shift_action = action
        space_pressed = space_action == 1 and not self.prev_space_held
        shift_held = shift_action == 1
        
        self.steps += 1

        # --- 1. Handle Player Input ---
        # Move cursor
        dx, dy = 0, 0
        if movement == 1: dy = -self.CURSOR_SPEED
        elif movement == 2: dy = self.CURSOR_SPEED
        elif movement == 3: dx = -self.CURSOR_SPEED
        elif movement == 4: dx = self.CURSOR_SPEED
        self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dx, 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dy, 0, self.HEIGHT)

        # Grab word (Shift is held)
        if shift_held and self.held_word_idx == -1:
            for i, word in enumerate(self.words_on_screen):
                if word['rect'].collidepoint(self.cursor_pos):
                    self.held_word_idx = i
                    # Find target definition for continuous reward
                    for d in self.definitions_on_screen:
                        if d['word'] == word['text']:
                            target_center = np.array(d['rect'].center)
                            current_dist = np.linalg.norm(word['pos'] - target_center)
                            self.dist_to_target_on_grab = current_dist
                            break
                    break
        
        # Ungrab if shift is released
        if not shift_held and self.held_word_idx != -1:
            self.held_word_idx = -1
            self.dist_to_target_on_grab = float('inf')

        # Release word (Space is pressed)
        if space_pressed and self.held_word_idx != -1:
            held_word = self.words_on_screen[self.held_word_idx]
            released_over_def = False
            for i, definition in enumerate(self.definitions_on_screen):
                if definition['rect'].collidepoint(held_word['pos']):
                    released_over_def = True
                    if definition['word'] == held_word['text']: # Correct match
                        # Event-based reward
                        bonus = 5 if held_word['pos'][1] < self.HEIGHT * 0.25 else 1
                        reward += bonus
                        self.score += bonus
                        self.matched_words_count += 1
                        
                        # Visuals & Sound
                        self._create_particles(held_word['pos'], self.COLOR_CORRECT, 30)

                        # Remove matched items
                        self.words_on_screen.pop(self.held_word_idx)
                        self.definitions_on_screen.pop(i)
                        
                        # Spawn new pair and increase difficulty
                        self._spawn_new_word_definition_pair()
                        if self.matched_words_count % 2 == 0:
                            self.word_fall_speed = min(20.0, self.word_fall_speed + 0.1)

                    else: # Incorrect match
                        reward -= 1
                        self.score -= 1
                        self._create_particles(held_word['pos'], self.COLOR_INCORRECT, 20)
                    
                    self.held_word_idx = -1
                    self.dist_to_target_on_grab = float('inf')
                    break
            
            if not released_over_def: # Released in empty space
                self.held_word_idx = -1
                self.dist_to_target_on_grab = float('inf')

        # --- 2. Update Game State ---
        # Move held word
        if self.held_word_idx != -1:
            word = self.words_on_screen[self.held_word_idx]
            
            # Continuous reward calculation
            target_center = None
            for d in self.definitions_on_screen:
                if d['word'] == word['text']:
                    target_center = np.array(d['rect'].center)
                    break
            
            if target_center is not None:
                prev_dist = np.linalg.norm(word['pos'] - target_center)
                word['pos'][:] = self.cursor_pos
                new_dist = np.linalg.norm(word['pos'] - target_center)
                
                if new_dist < prev_dist:
                    reward += 0.1
                else:
                    reward -= 0.1

        # Update falling words and check for misses
        indices_to_remove = []
        for i, word in enumerate(self.words_on_screen):
            if i != self.held_word_idx:
                word['pos'][1] += self.word_fall_speed
            
            word['rect'].center = word['pos']

            if word['pos'][1] > self.HEIGHT + word['rect'].height / 2:
                indices_to_remove.append(i)
        
        # Process misses
        for i in sorted(indices_to_remove, reverse=True):
            missed_word_obj = self.words_on_screen.pop(i)
            reward -= 2
            self.score -= 2
            self.missed_words += 1
            self._create_particles(missed_word_obj['pos'], self.COLOR_INCORRECT, 15)

            # Remove corresponding definition and spawn a new pair
            for j, d in enumerate(self.definitions_on_screen):
                if d['word'] == missed_word_obj['text']:
                    self.definitions_on_screen.pop(j)
                    break
            self._spawn_new_word_definition_pair()

            # Adjust held_word_idx if necessary
            if self.held_word_idx != -1:
                if i == self.held_word_idx: self.held_word_idx = -1
                elif i < self.held_word_idx: self.held_word_idx -= 1

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1

        # --- 3. Check Termination ---
        terminated = False
        if self.matched_words_count >= self.WIN_MATCHES:
            reward += 100
            self.score += 100
            self.game_over = True
            self.win = True
            terminated = True
        elif self.missed_words >= self.LOSE_MISSES:
            reward -= 100
            self.score -= 100
            self.game_over = True
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        # Update previous action state
        self.prev_space_held = space_action == 1
        self.prev_shift_held = shift_held

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _render_game(self):
        # Render definition boxes
        for d in self.definitions_on_screen:
            pygame.draw.rect(self.screen, self.COLOR_DEF_BOX, d['rect'], border_radius=8)
            pygame.draw.rect(self.screen, self.COLOR_DEF_TEXT, d['rect'], width=1, border_radius=8)
            
            # Word wrap for definition text
            words = d['text'].split(' ')
            lines = []
            current_line = ""
            for word in words:
                test_line = current_line + word + " "
                if self.font_def.size(test_line)[0] < d['rect'].width - 10:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word + " "
            lines.append(current_line)
            
            for i, line in enumerate(lines):
                text_surf = self.font_def.render(line, True, self.COLOR_DEF_TEXT)
                text_rect = text_surf.get_rect(centerx=d['rect'].centerx, y=d['rect'].top + 5 + i * 18)
                self.screen.blit(text_surf, text_rect)

        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(p['life'] * (255 / 20))))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((6, 6), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, 3, 3, 3, color)
            self.screen.blit(temp_surf, (int(p['pos'][0]) - 3, int(p['pos'][1]) - 3))
            
        # Render falling words
        for i, word in enumerate(self.words_on_screen):
            if i == self.held_word_idx: # Draw glow for held word
                glow_surf = pygame.Surface((word['rect'].width + 20, word['rect'].height + 20), pygame.SRCALPHA)
                pygame.draw.ellipse(glow_surf, self.COLOR_HELD_GLOW, glow_surf.get_rect())
                self.screen.blit(glow_surf, (word['rect'].centerx - glow_surf.get_width()//2, word['rect'].centery - glow_surf.get_height()//2))
            self.screen.blit(word['surf'], word['rect'])

        # Render cursor
        x, y = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        pygame.gfxdraw.aacircle(self.screen, x, y, 10, self.COLOR_CURSOR)
        pygame.gfxdraw.aacircle(self.screen, x, y, 9, self.COLOR_CURSOR)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (x-5, y), (x+5, y), 1)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (x, y-5), (x, y+5), 1)

    def _render_ui(self):
        # Score
        score_text = f"Score: {self.score}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Misses
        miss_text = f"Misses: {self.missed_words} / {self.LOSE_MISSES}"
        miss_surf = self.font_ui.render(miss_text, True, self.COLOR_INCORRECT)
        self.screen.blit(miss_surf, (self.WIDTH - miss_surf.get_width() - 10, 10))
        
        # Matches
        match_text = f"Matches: {self.matched_words_count} / {self.WIN_MATCHES}"
        match_surf = self.font_ui.render(match_text, True, self.COLOR_CORRECT)
        self.screen.blit(match_surf, (self.WIDTH - match_surf.get_width() - 10, 30))

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_CORRECT if self.win else self.COLOR_INCORRECT
            
            end_surf = self.font_game_over.render(message, True, color)
            end_rect = end_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_surf, end_rect)

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "matches": self.matched_words_count,
            "misses": self.missed_words,
            "fall_speed": self.word_fall_speed,
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set the video driver to a dummy one for headless execution
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    env = GameEnv(render_mode="rgb_array")
    
    # --- To actually see the game window, comment out the os.environ line ---
    # --- and use this block instead of the headless one below.           ---
    # env = GameEnv(render_mode="rgb_array")
    # screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    # pygame.display.set_caption("Word Drop")
    # obs, info = env.reset()
    # done = False
    # while not done:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             done = True
        
    #     # Simple keyboard mapping for human play
    #     keys = pygame.key.get_pressed()
    #     move = 0 # none
    #     if keys[pygame.K_UP]: move = 1
    #     elif keys[pygame.K_DOWN]: move = 2
    #     elif keys[pygame.K_LEFT]: move = 3
    #     elif keys[pygame.K_RIGHT]: move = 4
        
    #     space = 1 if keys[pygame.K_SPACE] else 0
    #     shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
    #     action = [move, space, shift]
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     done = terminated or truncated

    #     # Display the observation
    #     frame = np.transpose(obs, (1, 0, 2))
    #     surf = pygame.surfarray.make_surface(frame)
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()

    #     if info['steps'] % 30 == 0:
    #         print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

    # env.close()

    # Headless test loop
    print("Running headless test...")
    obs, info = env.reset()
    total_reward = 0
    for i in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if (i+1) % 100 == 0:
            print(f"Step {i+1}: Last reward={reward:.2f}, Total reward={total_reward:.2f}, Info={info}")
        if terminated or truncated:
            print("Episode finished.")
            obs, info = env.reset()
            total_reward = 0
    env.close()
    print("Headless test complete.")