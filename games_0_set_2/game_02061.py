
# Generated: 2025-08-28T03:34:09.508805
# Source Brief: brief_02061.md
# Brief Index: 2061

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to grab a word, and Shift to release it over a definition."
    )

    game_description = (
        "A fast-paced vocabulary puzzle. Drag falling words to their matching definitions at the bottom of the screen before time runs out. Match 20 words to win!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        self.fps = 30

        self.observation_space = Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # --- Visuals & Colors ---
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_DEF_BOX = (45, 55, 65)
        self.COLOR_DEF_TEXT = (180, 190, 200)
        self.COLOR_CURSOR = (255, 200, 0)
        self.COLOR_CORRECT = (0, 255, 150)
        self.COLOR_INCORRECT = (255, 80, 80)
        self.COLOR_LINE = (255, 200, 0, 150)

        self.font_word = pygame.font.SysFont("Consolas", 22, bold=True)
        self.font_def = pygame.font.SysFont("Arial", 14)
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_feedback = pygame.font.SysFont("Arial", 20, bold=True)
        
        self.word_data = self._get_word_data()
        self.win_condition = 20
        self.max_time = 60 * self.fps # 60 seconds

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.timer = 0
        self.game_over = False
        self.win = False
        self.cursor_pos = [0, 0]
        self.previous_cursor_pos = [0, 0]
        self.held_word = None
        self.previous_space_held = False
        self.previous_shift_held = False
        self.fall_speed = 0.0
        self.words_on_screen = []
        self.definitions_on_screen = []
        self.round_word_indices = []
        self.available_def_indices = set()
        self.particles = []
        self.feedback_items = []
        self.word_spawn_timer = 0

        self.validate_implementation()
        self.reset()

    def _get_word_data(self):
        return [
            ("AGILE", "Able to move quickly and easily."),
            ("BRAVE", "Ready to face and endure danger or pain."),
            ("CALM", "Not showing or feeling nervousness or anger."),
            ("DREAM", "A series of thoughts, images, and sensations."),
            ("ECHO", "A sound caused by the reflection of sound waves."),
            ("FABLE", "A short story, typically with animals as characters."),
            ("GIANT", "Of very great size or force."),
            ("HABIT", "A settled or regular tendency or practice."),
            ("IGLOO", "A dome-shaped Eskimo house, typically of snow."),
            ("JOKER", "A person who is fond of joking."),
            ("KAYAK", "A canoe of a type used originally by the Inuit."),
            ("LEMON", "A yellow, oval citrus fruit with thick skin."),
            ("MAGIC", "The power of apparently influencing events."),
            ("NOBLE", "Belonging to a hereditary class with high status."),
            ("OCEAN", "A very large expanse of sea."),
            ("PIXEL", "A minute area of illumination on a display screen."),
            ("QUERY", "A question, especially one addressed to an official."),
            ("ROBOT", "A machine capable of carrying out complex actions."),
            ("SILENT", "Not making or accompanied by any sound."),
            ("TASTY", "Having a pleasant, distinct flavor."),
            ("UNITY", "The state of being united or joined as a whole."),
            ("VIVID", "Producing powerful feelings or strong, clear images."),
            ("WALTZ", "A dance in triple time performed by a couple."),
            ("XENON", "The chemical element of atomic number 54."),
            ("YACHT", "A medium-sized sailboat equipped for cruising or racing."),
            ("ZEBRA", "An African wild horse with black-and-white stripes."),
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.timer = self.max_time
        self.game_over = False
        self.win = False
        
        self.cursor_pos = [self.width // 2, self.height // 2]
        self.previous_cursor_pos = list(self.cursor_pos)
        self.held_word = None
        self.previous_space_held = False
        self.previous_shift_held = False
        
        self.fall_speed = 30.0 / self.fps  # 30 pixels per second
        
        self.words_on_screen = []
        self.definitions_on_screen = []
        self.particles = []
        self.feedback_items = []
        
        self._setup_puzzle()
        
        return self._get_observation(), self._get_info()

    def _setup_puzzle(self):
        # Select words for this round
        all_indices = list(range(len(self.word_data)))
        self.round_word_indices = self.np_random.choice(all_indices, self.win_condition, replace=False).tolist()
        
        # Choose initial definitions for the bottom
        num_defs = 4
        initial_def_indices = self.np_random.choice(self.round_word_indices, num_defs, replace=False).tolist()
        self.available_def_indices = set(self.round_word_indices) - set(initial_def_indices)
        
        self.definitions_on_screen = []
        box_width = self.width // num_defs
        for i, def_idx in enumerate(initial_def_indices):
            word, definition_text = self.word_data[def_idx]
            rect = pygame.Rect(i * box_width, self.height - 50, box_width, 50)
            self.definitions_on_screen.append({
                "text": definition_text, "def_index": def_idx, "rect": rect, "flash_timer": 0
            })
            
        # Spawn initial words
        self.word_spawn_timer = 0
        for _ in range(3):
            self._spawn_word()

    def _spawn_word(self):
        if not self.definitions_on_screen: return
        
        target_def = self.np_random.choice(self.definitions_on_screen)
        def_idx = target_def["def_index"]
        word_text, _ = self.word_data[def_idx]
        
        text_surface = self.font_word.render(word_text, True, self.COLOR_TEXT)
        x = self.np_random.integers(10, self.width - text_surface.get_width() - 10)
        y = -20.0
        
        self.words_on_screen.append({
            "text": word_text, "surface": text_surface, "pos": [x, y], "def_index": def_idx
        })
        self.word_spawn_timer = self.np_random.integers(int(2.5 * self.fps), int(4 * self.fps))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        is_space_press = space_held and not self.previous_space_held
        is_shift_press = shift_held and not self.previous_shift_held

        self.previous_cursor_pos = list(self.cursor_pos)
        self._update_cursor(movement)

        if is_space_press and not self.held_word:
            self._grab_word()
            
        if is_shift_press and self.held_word:
            match_reward = self._release_word()
            reward += match_reward
        
        if self.held_word:
            reward += self._calculate_continuous_reward()

        self._update_world_state()
        
        self.timer -= 1
        self.steps += 1

        if self.steps > 0 and self.steps % (10 * self.fps) == 0:
            self.fall_speed += (0.05 * self.fps) / self.fps # 0.05 px/sec increase every 10 sec

        terminated = False
        if self.timer <= 0:
            self.game_over = True
            terminated = True
            reward -= 100.0
        elif self.score >= self.win_condition:
            self.game_over = True
            self.win = True
            terminated = True
            reward += 50.0 + (self.timer / self.fps) # Win bonus + time bonus

        self.previous_space_held = space_held
        self.previous_shift_held = shift_held
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_cursor(self, movement):
        cursor_speed = 10
        if movement == 1: self.cursor_pos[1] -= cursor_speed
        elif movement == 2: self.cursor_pos[1] += cursor_speed
        elif movement == 3: self.cursor_pos[0] -= cursor_speed
        elif movement == 4: self.cursor_pos[0] += cursor_speed
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.width)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.height)

    def _grab_word(self):
        # sound: grab_word.wav
        grab_radius = 30
        closest_word = None
        min_dist = float('inf')
        
        for word in self.words_on_screen:
            word_rect = word["surface"].get_rect(topleft=word["pos"])
            dist = math.hypot(word_rect.centerx - self.cursor_pos[0], word_rect.centery - self.cursor_pos[1])
            if dist < grab_radius and dist < min_dist:
                min_dist = dist
                closest_word = word
        
        if closest_word:
            self.held_word = closest_word
            self.words_on_screen.remove(closest_word)

    def _release_word(self):
        # sound: release_word.wav
        dropped_on_target = False
        for i, definition in enumerate(self.definitions_on_screen):
            if definition["rect"].collidepoint(self.cursor_pos):
                dropped_on_target = True
                # Correct Match
                if self.held_word["def_index"] == definition["def_index"]:
                    # sound: correct_match.wav
                    self.score += 1
                    self._create_particles(definition["rect"].center, self.COLOR_CORRECT)
                    self._add_feedback("Matched!", definition["rect"].center, self.COLOR_CORRECT)
                    self._replace_definition(i)
                    self.held_word = None
                    return 10.0
                # Incorrect Match
                else:
                    # sound: incorrect_match.wav
                    definition["flash_timer"] = 15 # frames
                    self._add_feedback("Wrong!", self.cursor_pos, self.COLOR_INCORRECT)
                    # Word goes back to falling
                    self.held_word["pos"] = self.cursor_pos
                    self.words_on_screen.append(self.held_word)
                    self.held_word = None
                    return -1.0
        
        # Dropped in empty space
        if not dropped_on_target:
            self.held_word["pos"] = self.cursor_pos
            self.words_on_screen.append(self.held_word)
            self.held_word = None
        
        return 0.0

    def _calculate_continuous_reward(self):
        if not self.held_word: return 0.0
        
        target_def = None
        for d in self.definitions_on_screen:
            if d["def_index"] == self.held_word["def_index"]:
                target_def = d
                break
        
        if not target_def: return 0.0

        dist_before = math.hypot(self.previous_cursor_pos[0] - target_def["rect"].centerx, self.previous_cursor_pos[1] - target_def["rect"].centery)
        dist_now = math.hypot(self.cursor_pos[0] - target_def["rect"].centerx, self.cursor_pos[1] - target_def["rect"].centery)
        
        if dist_now < dist_before:
            return 0.1
        elif dist_now > dist_before:
            return -0.1
        return 0.0

    def _replace_definition(self, index_to_replace):
        if not self.available_def_indices: return
        
        new_def_idx = self.np_random.choice(list(self.available_def_indices))
        self.available_def_indices.remove(new_def_idx)
        
        word, definition_text = self.word_data[new_def_idx]
        old_rect = self.definitions_on_screen[index_to_replace]["rect"]
        
        self.definitions_on_screen[index_to_replace] = {
            "text": definition_text, "def_index": new_def_idx, "rect": old_rect, "flash_timer": 0
        }

    def _update_world_state(self):
        # Update falling words
        for word in self.words_on_screen[:]:
            word["pos"][1] += self.fall_speed
            if word["pos"][1] > self.height:
                self.words_on_screen.remove(word)
        
        # Update held word position
        if self.held_word:
            self.held_word["pos"] = self.cursor_pos
            
        # Spawn new words
        self.word_spawn_timer -= 1
        if self.word_spawn_timer <= 0:
            self._spawn_word()

        # Update particles
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
                
        # Update feedback text
        for item in self.feedback_items[:]:
            item["pos"][1] -= 0.5
            item["life"] -= 1
            if item["life"] <= 0:
                self.feedback_items.remove(item)
        
        # Update definition box flash
        for d in self.definitions_on_screen:
            if d["flash_timer"] > 0:
                d["flash_timer"] -= 1

    def _create_particles(self, pos, color):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({"pos": list(pos), "vel": vel, "life": life, "color": color})
    
    def _add_feedback(self, text, pos, color):
        self.feedback_items.append({
            "text": text, "pos": list(pos), "color": color, "life": self.fps
        })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render definitions
        for d in self.definitions_on_screen:
            color = self.COLOR_INCORRECT if d["flash_timer"] > 0 else self.COLOR_DEF_BOX
            pygame.draw.rect(self.screen, color, d["rect"], border_radius=5)
            pygame.draw.rect(self.screen, tuple(c*1.5 for c in color[:3]), d["rect"], width=1, border_radius=5)
            
            # Word wrapping for definitions
            words = d["text"].split(' ')
            lines = []
            current_line = ""
            for word in words:
                test_line = current_line + word + " "
                if self.font_def.size(test_line)[0] < d["rect"].width - 20:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word + " "
            lines.append(current_line)

            for i, line in enumerate(lines):
                text_surf = self.font_def.render(line, True, self.COLOR_DEF_TEXT)
                text_rect = text_surf.get_rect(centerx=d["rect"].centerx, y=d["rect"].top + 5 + i * 15)
                self.screen.blit(text_surf, text_rect)

        # Render falling words
        for word in self.words_on_screen:
            self.screen.blit(word["surface"], word["pos"])
        
        # Render held word
        if self.held_word:
            self.screen.blit(self.held_word["surface"], self.held_word["surface"].get_rect(center=self.cursor_pos))
        
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 30.0))
            color = (*p["color"], alpha)
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, color)

        # Render feedback text
        for item in self.feedback_items:
            alpha = min(255, int(255 * (item["life"] / (self.fps/2))))
            if item["life"] < self.fps/2:
                alpha = int(255 * (item["life"] / (self.fps/2)))
            
            text_surf = self.font_feedback.render(item["text"], True, item["color"])
            text_surf.set_alpha(alpha)
            text_rect = text_surf.get_rect(center=item["pos"])
            self.screen.blit(text_surf, text_rect)
            
        # Render cursor
        pygame.draw.circle(self.screen, self.COLOR_CURSOR, [int(c) for c in self.cursor_pos], 8, 2)
        pygame.draw.circle(self.screen, self.COLOR_CURSOR, [int(c) for c in self.cursor_pos], 2)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}/{self.win_condition}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = max(0, self.timer / self.fps)
        timer_color = self.COLOR_TEXT if time_left > 10 else self.COLOR_INCORRECT
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, timer_color)
        timer_rect = timer_text.get_rect(topright=(self.width - 10, 10))
        self.screen.blit(timer_text, timer_rect)

        # Game Over / Win Text
        if self.game_over:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "YOU WIN!" if self.win else "TIME'S UP!"
            end_text_color = self.COLOR_CORRECT if self.win else self.COLOR_INCORRECT
            end_text = self.font_ui.render(end_text_str, True, end_text_color)
            end_rect = end_text.get_rect(center=(self.width // 2, self.height // 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer / self.fps,
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # To run the game and play it manually
    import os
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption(env.game_description)
    
    obs, info = env.reset()
    terminated = False
    
    while not terminated:
        # --- Manual Control ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        # --- Render to screen ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        env.clock.tick(env.fps)

    env.close()
    print("Game Over!")
    print(f"Final Score: {info['score']}")