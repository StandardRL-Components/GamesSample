import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:16:13.138160
# Source Brief: brief_00372.md
# Brief Index: 372
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
        "Become a tiny detective, collecting clues that require different 'sizes' to interact with. "
        "Match clue pairs to build your case, and finally, enter the courtroom to accuse the correct culprit."
    )
    user_guide = (
        "Use arrow keys to move. Press shift to change your size. "
        "Press space to collect clues, match them, and make your final accusation."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1000
        self.NUM_CLUE_PAIRS = 3
        self.NUM_SUSPECTS = 3

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

        # Visuals
        self._init_colors()
        self._init_fonts()

        # Player state
        self.player_pos = np.array([0.0, 0.0])
        self.player_size_level = 0
        self.player_sizes = [10, 15, 20] # Small, Medium, Large radii
        self.player_speed = 10.0
        
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.phase = "exploration" # exploration, court
        self.clues = []
        self.suspects = []
        self.culprit_id = -1
        self.collected_clues = []
        
        # Interaction state
        self.matching_state = {
            "active": False,
            "clue1_id": None,
            "selected_clue2_idx": 0,
        }
        self.court_selection_idx = 0
        self.previous_space_held = False
        self.previous_shift_held = False

        # Effects
        self.particles = []
        self.feedback_effect = None

        # Initialize state variables by calling reset
        # self.reset() is called by the environment wrapper, so not needed here.

    def _init_colors(self):
        self.COLOR_BG = (20, 15, 40) # Dark purple
        self.COLOR_WALL = (60, 50, 90)
        self.COLOR_PLAYER = (255, 220, 0) # Bright Yellow
        self.COLOR_PLAYER_GLOW = (255, 220, 0, 50)
        self.COLOR_CLUE_SPARKLE = (0, 255, 255) # Cyan
        self.COLOR_CLUE_COLLECTED = (100, 100, 120)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_SUCCESS = (0, 255, 100)
        self.COLOR_FAIL = (255, 50, 50)
        self.COLOR_UI_BG = (40, 30, 70, 180)
        self.COLOR_HIGHLIGHT = (255, 255, 255)

    def _init_fonts(self):
        self.font_main = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 36, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 14)
    
    def _generate_case(self):
        self.clues = []
        self.suspects = []
        
        # Define clue pairs
        clue_data = [
            ("MUDDY FOOTPRINT", "TORN FABRIC SWATCH"),
            ("MYSTERIOUS SYMBOL", "CIPHERED NOTE"),
            ("VICTIM'S SCHEDULE", "SUSPICIOUS ALIBI")
        ]
        
        # Define suspects
        suspect_names = ["MR. GREEN", "MS. SCARLETT", "COL. MUSTARD"]
        self.culprit_id = self.np_random.integers(0, self.NUM_SUSPECTS)

        for i in range(self.NUM_SUSPECTS):
            self.suspects.append({"id": i, "name": suspect_names[i], "is_culprit": i == self.culprit_id})

        # Generate clues
        for i in range(self.NUM_CLUE_PAIRS):
            pos1 = (self.np_random.uniform(50, self.WIDTH - 50), self.np_random.uniform(50, self.HEIGHT - 100))
            pos2 = (self.np_random.uniform(50, self.WIDTH - 50), self.np_random.uniform(50, self.HEIGHT - 100))
            
            size_req1 = self.np_random.integers(0, 3)
            size_req2 = self.np_random.integers(0, 3)

            desc1, desc2 = clue_data[i]
            if i == self.NUM_CLUE_PAIRS - 1: # Final clue pair points to the culprit
                desc2 = f"ALIBI OF {self.suspects[self.culprit_id]['name']}"

            self.clues.append({
                "id": i * 2, "pair_id": i, "pos": pos1, "size_req": size_req1, 
                "collected": False, "matched": False, "desc": desc1
            })
            self.clues.append({
                "id": i * 2 + 1, "pair_id": i, "pos": pos2, "size_req": size_req2,
                "collected": False, "matched": False, "desc": desc2
            })
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = np.array([self.WIDTH / 2.0, self.HEIGHT / 2.0])
        self.player_size_level = 1 # Start at medium size
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.phase = "exploration"
        
        self._generate_case()
        self.collected_clues = []
        
        self.matching_state = {"active": False, "clue1_id": None, "selected_clue2_idx": 0}
        self.court_selection_idx = 0
        self.previous_space_held = False
        self.previous_shift_held = False
        
        self.particles = []
        self.feedback_effect = None
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.previous_space_held
        shift_pressed = shift_held and not self.previous_shift_held

        # --- Handle Input and State Changes ---
        # Movement
        move_vec = np.array([0.0, 0.0])
        if movement == 1: move_vec[1] = -1.0 # Up
        elif movement == 2: move_vec[1] = 1.0 # Down
        elif movement == 3: move_vec[0] = -1.0 # Left
        elif movement == 4: move_vec[0] = 1.0 # Right
        
        self.player_pos += move_vec * self.player_speed
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WIDTH)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.HEIGHT)
        
        # Phase-dependent actions
        if self.phase == "exploration":
            reward += self._handle_exploration_phase(shift_pressed, space_pressed)
        elif self.phase == "court":
            reward += self._handle_court_phase(movement, space_pressed)

        # --- Update Game World ---
        self._update_particles()
        if self.feedback_effect and self.feedback_effect['life'] > 0:
            self.feedback_effect['life'] -= 1
        else:
            self.feedback_effect = None

        # Check for phase transition
        if self.phase == "exploration" and len(self.collected_clues) == self.NUM_CLUE_PAIRS * 2:
            self.phase = "court"
            self._create_feedback_effect("COURT IS IN SESSION", self.COLOR_HIGHLIGHT, 90)

        self.previous_space_held = space_held
        self.previous_shift_held = shift_held
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over: # Ran out of time
            reward -= 50 # Penalty for running out of time
        
        self.score += reward
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_exploration_phase(self, shift_pressed, space_pressed):
        reward = 0
        
        # Cancel matching mode if player moves away from the first clue
        if self.matching_state["active"]:
            clue1 = self._get_clue_by_id(self.matching_state["clue1_id"])
            dist = np.linalg.norm(self.player_pos - np.array(clue1['pos']))
            if dist > self.player_sizes[self.player_size_level] + 10:
                self.matching_state["active"] = False
                self._create_feedback_effect("MATCH CANCELLED", self.COLOR_FAIL, 30)

        # Shift: Change size OR cycle matching clues
        if shift_pressed:
            if self.matching_state["active"]:
                # Cycle through other collected, unmatched clues
                other_clues = [c for c in self.clues if c['collected'] and not c['matched'] and c['id'] != self.matching_state['clue1_id']]
                if other_clues:
                    self.matching_state["selected_clue2_idx"] = (self.matching_state["selected_clue2_idx"] + 1) % len(other_clues)
            else:
                # Cycle player size
                self.player_size_level = (self.player_size_level + 1) % len(self.player_sizes)
        
        # Space: Interact with clues (collect or match)
        if space_pressed:
            clue_under_cursor = self._get_clue_at_player_pos()
            
            if self.matching_state["active"]:
                # Attempt to finalize a match
                other_clues = [c for c in self.clues if c['collected'] and not c['matched'] and c['id'] != self.matching_state['clue1_id']]
                if other_clues:
                    clue1 = self._get_clue_by_id(self.matching_state["clue1_id"])
                    clue2 = other_clues[self.matching_state["selected_clue2_idx"]]
                    if clue1['pair_id'] == clue2['pair_id']:
                        reward += 5
                        clue1['matched'] = True
                        clue2['matched'] = True
                        self._create_feedback_effect("MATCH FOUND!", self.COLOR_SUCCESS, 60)
                    else:
                        reward -= 1
                        self._create_feedback_effect("INCORRECT MATCH", self.COLOR_FAIL, 60)
                self.matching_state["active"] = False
            elif clue_under_cursor:
                clue = clue_under_cursor
                if not clue['collected']:
                    # Collect clue
                    if self.player_size_level == clue['size_req']:
                        clue['collected'] = True
                        self.collected_clues.append(clue['id'])
                        reward += 1
                        self._create_particles(clue['pos'], self.COLOR_CLUE_SPARKLE, 20)
                    else:
                        self._create_feedback_effect("WRONG SIZE", self.COLOR_FAIL, 30)
                elif not clue['matched']:
                    # Initiate matching
                    self.matching_state["active"] = True
                    self.matching_state["clue1_id"] = clue['id']
                    self.matching_state["selected_clue2_idx"] = 0
                    self._create_feedback_effect("MATCHING...", self.COLOR_HIGHLIGHT, 30)
        return reward

    def _handle_court_phase(self, movement, space_pressed):
        reward = 0
        # Movement actions 3 (left) and 4 (right) cycle through suspects
        if movement == 3: # Left
            self.court_selection_idx = (self.court_selection_idx - 1 + self.NUM_SUSPECTS) % self.NUM_SUSPECTS
        elif movement == 4: # Right
            self.court_selection_idx = (self.court_selection_idx + 1) % self.NUM_SUSPECTS
        
        if space_pressed:
            accused_suspect = self.suspects[self.court_selection_idx]
            if accused_suspect['is_culprit']:
                reward += 100
                self._create_feedback_effect("CASE CLOSED!", self.COLOR_SUCCESS, 120)
            else:
                reward -= 100
                self._create_feedback_effect("WRONG ACCUSATION!", self.COLOR_FAIL, 120)
            self.game_over = True
        return reward
    
    def _get_clue_at_player_pos(self):
        player_radius = self.player_sizes[self.player_size_level]
        for clue in self.clues:
            dist = np.linalg.norm(self.player_pos - np.array(clue['pos']))
            if dist < player_radius + 5: # 5 is clue radius
                return clue
        return None

    def _get_clue_by_id(self, clue_id):
        for clue in self.clues:
            if clue['id'] == clue_id:
                return clue
        return None

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                'pos': np.array(pos, dtype=float),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'life': self.np_random.integers(15, 30),
                'color': color
            })

    def _create_feedback_effect(self, text, color, life):
        self.feedback_effect = {"text": text, "color": color, "life": life}
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render clues
        for clue in self.clues:
            color = self.COLOR_CLUE_SPARKLE
            if clue['collected']:
                color = self.COLOR_CLUE_COLLECTED
            
            pygame.draw.circle(self.screen, color, (int(clue['pos'][0]), int(clue['pos'][1])), 5)
            
            if not clue['collected']: # Sparkle effect
                if self.np_random.random() < 0.1:
                    pygame.gfxdraw.filled_circle(
                        self.screen, int(clue['pos'][0]), int(clue['pos'][1]),
                        self.np_random.integers(6, 10), (*self.COLOR_CLUE_SPARKLE, 100)
                    )
            
            # Show size requirement
            size_color = [self.COLOR_FAIL, self.COLOR_TEXT, self.COLOR_SUCCESS][clue['size_req']]
            pygame.draw.rect(self.screen, size_color, (clue['pos'][0] - 5, clue['pos'][1] + 8, 10, 2))

        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 30.0))))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 2, (*p['color'], alpha))

        # Render player
        player_radius = self.player_sizes[self.player_size_level]
        pos_int = (int(self.player_pos[0]), int(self.player_pos[1]))
        
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(player_radius * 1.5), self.COLOR_PLAYER_GLOW)
        # Body
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, pos_int, player_radius)

    def _render_ui(self):
        # --- Top Bar ---
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.font_main.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

        # --- Bottom Bar (Clue Inventory) ---
        bar_h = 60
        s = pygame.Surface((self.WIDTH, bar_h), pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        
        matched_pairs = sum(1 for i in range(self.NUM_CLUE_PAIRS) if self._get_clue_by_id(i*2)['matched'])
        progress_w = (self.WIDTH - 20) * (matched_pairs / self.NUM_CLUE_PAIRS) if self.NUM_CLUE_PAIRS > 0 else 0
        
        pygame.draw.rect(s, self.COLOR_WALL, (10, 10, self.WIDTH - 20, 10))
        if progress_w > 0:
            pygame.draw.rect(s, self.COLOR_SUCCESS, (10, 10, progress_w, 10))
        
        # Draw collected clue icons
        x_offset = 15
        for clue_id in sorted(self.collected_clues):
            clue = self._get_clue_by_id(clue_id)
            color = self.COLOR_SUCCESS if clue['matched'] else self.COLOR_HIGHLIGHT
            pygame.draw.rect(s, color, (x_offset, 30, 8, 20))
            pygame.draw.rect(s, self.COLOR_BG, (x_offset+2, 32, 4, 16))
            x_offset += 15
        self.screen.blit(s, (0, self.HEIGHT - bar_h))

        # --- Phase Specific UI ---
        if self.phase == "court":
            self._render_court_phase()
        elif self.matching_state["active"]:
            self._render_matching_ui()

        # --- Central Feedback Text ---
        if self.feedback_effect:
            alpha = max(0, min(255, int(255 * (self.feedback_effect['life'] / 30.0))))
            text_surf = self.font_large.render(self.feedback_effect['text'], True, (*self.feedback_effect['color'], alpha))
            pos = (self.WIDTH // 2 - text_surf.get_width() // 2, self.HEIGHT // 2 - text_surf.get_height() // 2)
            self.screen.blit(text_surf, pos)

    def _render_matching_ui(self):
        clue1 = self._get_clue_by_id(self.matching_state['clue1_id'])
        other_clues = [c for c in self.clues if c['collected'] and not c['matched'] and c['id'] != clue1['id']]
        
        clue2_text = "..."
        if other_clues:
            clue2 = other_clues[self.matching_state['selected_clue2_idx']]
            clue2_text = clue2['desc']

        text1 = self.font_small.render(f"MATCH: {clue1['desc']}", True, self.COLOR_HIGHLIGHT)
        text2 = self.font_small.render(f"WITH: {clue2_text}", True, self.COLOR_CLUE_SPARKLE)
        
        ui_pos_y = self.HEIGHT - 85
        self.screen.blit(text1, (15, ui_pos_y))
        self.screen.blit(text2, (15, ui_pos_y + 15))

    def _render_court_phase(self):
        s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        s.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 220))
        
        title = self.font_large.render("WHO IS THE CULPRIT?", True, self.COLOR_TEXT)
        s.blit(title, (self.WIDTH // 2 - title.get_width() // 2, 50))

        suspect_w = 150
        spacing = (self.WIDTH - self.NUM_SUSPECTS * suspect_w) / (self.NUM_SUSPECTS + 1)
        
        for i, suspect in enumerate(self.suspects):
            x = spacing + i * (suspect_w + spacing)
            y = 150
            
            box_color = self.COLOR_HIGHLIGHT if i == self.court_selection_idx else self.COLOR_WALL
            pygame.draw.rect(s, box_color, (x, y, suspect_w, 150), 3)
            
            name_surf = self.font_main.render(suspect['name'], True, self.COLOR_TEXT)
            s.blit(name_surf, (x + suspect_w // 2 - name_surf.get_width() // 2, y + 60))
        
        self.screen.blit(s, (0, 0))
    
    def _get_info(self):
        matched_pairs = sum(1 for i in range(self.NUM_CLUE_PAIRS) if self._get_clue_by_id(i*2)['matched'])
        return {
            "score": self.score,
            "steps": self.steps,
            "phase": self.phase,
            "collected_clues": len(self.collected_clues),
            "matched_pairs": matched_pairs,
        }
    
    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # For manual play, we need a real display
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tiny Detective")
    clock = pygame.time.Clock()
    
    terminated = False
    
    while not terminated:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, term, trunc, info = env.step(action)
        terminated = term or trunc
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward}, Score: {info['score']}, Phase: {info['phase']}")

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    print(f"Game Over! Final Score: {info['score']}")
    env.close()