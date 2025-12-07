import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:13:12.287740
# Source Brief: brief_00956.md
# Brief Index: 956
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a surreal murder mystery game.
    The agent controls a cursor to explore dreamscapes, find clues,
    and solve a murder.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Explore surreal dreamscapes as a detective to find clues, interact with witnesses, "
        "and solve a bizarre murder mystery."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to interact with objects. "
        "During the accusation phase, press shift to cycle suspects and space to confirm."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    CURSOR_SPEED = 15
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG_STUDY = (10, 20, 40)
    COLOR_BG_GARDEN = (20, 40, 10)
    COLOR_BG_BALLROOM = (40, 10, 20)
    COLOR_CURSOR = (50, 255, 255)
    COLOR_CURSOR_GLOW = (50, 255, 255, 50)
    COLOR_CLUE = (50, 255, 100)
    COLOR_CLUE_COLLECTED = (80, 120, 90)
    COLOR_PORTAL = (150, 50, 255)
    COLOR_WITNESS = (60, 60, 80)
    COLOR_TEXT = (230, 230, 240)
    COLOR_TEXT_BG = (10, 10, 20, 200)
    COLOR_ACCUSE_UI_BG = (20, 20, 30, 220)
    COLOR_ACCUSE_SELECTED = (255, 200, 50)
    COLOR_RED_HINT = (255, 80, 80)
    COLOR_GREEN_HINT = (80, 255, 80)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("sans-serif", 20)
        self.font_title = pygame.font.SysFont("serif", 28, bold=True)
        self.font_feedback = pygame.font.SysFont("sans-serif", 18)

        # --- Game Data ---
        self._define_game_data()

        # --- State Variables ---
        # These are initialized properly in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [0, 0]
        self.current_location_id = 'study'
        self.collected_clue_ids = set()
        self.visited_location_ids = set()
        self.game_phase = 'EXPLORATION'
        self.accusation_index = 0
        self.previous_space_held = False
        self.previous_shift_held = False
        self.feedback_queue = [] # Stores {'text', 'timer', 'color'} dicts
        self.anim_tick = 0
        
        # --- Final setup ---
        # self.reset() is called by the environment wrapper
        # self.validate_implementation() # Uncomment for self-testing

    def _define_game_data(self):
        """Defines the static content of the game like locations, clues, etc."""
        self.suspects = ["The Butler", "The Heiress", "The Rival"]
        self.murderer = "The Rival"
        self.all_clue_ids = {'clue_contract', 'clue_mud', 'clue_cufflink', 'clue_lie'}

        self.locations = {
            'study': {
                'name': "The Baron's Study",
                'bg_color': self.COLOR_BG_STUDY,
                'hotspots': [
                    {'id': 'clue_contract', 'type': 'clue', 'rect': pygame.Rect(100, 250, 50, 30), 'text': "A torn business contract with the Rival's name on it."},
                    {'id': 'clue_mud', 'type': 'clue', 'rect': pygame.Rect(500, 350, 60, 20), 'text': "A muddy bootprint. It could belong to anyone."},
                    {'id': 'portal_garden', 'type': 'portal', 'pos': (580, 200), 'radius': 40, 'target': 'garden', 'req': set()},
                    {'id': 'witness_butler', 'type': 'witness', 'rect': pygame.Rect(310, 150, 40, 100), 'text': "Butler: 'I saw the Rival arguing with the Baron just hours before... they were shouting about money.'"}
                ]
            },
            'garden': {
                'name': "The Wilted Garden",
                'bg_color': self.COLOR_BG_GARDEN,
                'hotspots': [
                    {'id': 'clue_cufflink', 'type': 'clue', 'rect': pygame.Rect(150, 320, 20, 20), 'text': "A silver cufflink, initialed 'R'. It must be the Rival's."},
                    {'id': 'portal_study', 'type': 'portal', 'pos': (60, 200), 'radius': 40, 'target': 'study', 'req': set()},
                    {'id': 'portal_ballroom', 'type': 'portal', 'pos': (580, 200), 'radius': 40, 'target': 'ballroom', 'req': {'clue_contract'}, 'locked_text': "The way is hazy. A key detail feels missing..."},
                    {'id': 'witness_heiress', 'type': 'witness', 'rect': pygame.Rect(400, 170, 40, 100), 'text': "Heiress: 'I was in my chambers all evening, I heard nothing. It's a tragedy.'"}
                ]
            },
            'ballroom': {
                'name': "The Silent Ballroom",
                'bg_color': self.COLOR_BG_BALLROOM,
                'hotspots': [
                     {'id': 'clue_lie', 'type': 'clue', 'rect': pygame.Rect(450, 300, 40, 40), 'text': "A single, misplaced dance card. The Heiress's name is on it... for the last dance. She wasn't in her room."},
                     {'id': 'portal_garden', 'type': 'portal', 'pos': (60, 200), 'radius': 40, 'target': 'garden', 'req': set()}
                ]
            }
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2]
        self.current_location_id = 'study'
        self.collected_clue_ids = set()
        self.visited_location_ids = {'study'}
        self.game_phase = 'EXPLORATION'
        self.accusation_index = 0
        self.previous_space_held = False
        self.previous_shift_held = False
        self.feedback_queue.clear()
        self.anim_tick = 0
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.anim_tick += 1
        reward = -0.01  # Small penalty for taking a step
        
        # --- Update feedback timers ---
        self.feedback_queue = [f for f in self.feedback_queue if f['timer'] > 0]
        for f in self.feedback_queue:
            f['timer'] -= 1

        # --- Unpack and process actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.previous_space_held
        shift_press = shift_held and not self.previous_shift_held

        # 1. Handle Movement
        if self.game_phase != 'ACCUSATION':
            if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
            elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
            elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
            elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED
            self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT)
        else: # In accusation mode, movement can be used to cycle choices as well
            if movement == 1 or movement == 3: # Up or Left
                if shift_press: # To avoid rapid cycling from holding key
                    self.accusation_index = (self.accusation_index - 1 + len(self.suspects)) % len(self.suspects)
            elif movement == 2 or movement == 4: # Down or Right
                if shift_press:
                    self.accusation_index = (self.accusation_index + 1) % len(self.suspects)

        # 2. Handle Interaction (Space Press)
        if space_press:
            interaction_reward = self._handle_interaction()
            reward += interaction_reward
        
        # 3. Handle Accusation Cycling (Shift Press)
        if shift_press and self.game_phase == 'ACCUSATION':
            self.accusation_index = (self.accusation_index + 1) % len(self.suspects)
            # sfx: UI cycle sound

        self.previous_space_held = space_held
        self.previous_shift_held = shift_held
        
        # --- Update Game State ---
        self.steps += 1
        self.score += reward
        
        # Check for transition to accusation phase
        if self.game_phase == 'EXPLORATION' and self.collected_clue_ids.issuperset(self.all_clue_ids):
            self.game_phase = 'ACCUSATION'
            self._add_feedback("All clues found! Time to make an accusation.", 300, self.COLOR_GREEN_HINT)

        terminated = self.steps >= self.MAX_STEPS or self.game_over
        if self.steps >= self.MAX_STEPS and not self.game_over:
            reward -= 50 # Penalty for running out of time
            self.score -= 50
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_interaction(self):
        """Processes a space-press interaction."""
        cursor_point = tuple(self.cursor_pos)
        
        # --- Interaction in Accusation Phase ---
        if self.game_phase == 'ACCUSATION':
            # Simplified: any space press in this mode confirms accusation
            return self._make_accusation()

        # --- Interaction in Exploration Phase ---
        location_data = self.locations[self.current_location_id]
        for hotspot in reversed(location_data['hotspots']): # Reverse to check top-most first
            
            is_colliding = False
            if 'rect' in hotspot and hotspot['rect'].collidepoint(cursor_point):
                is_colliding = True
            elif 'pos' in hotspot and math.hypot(cursor_point[0] - hotspot['pos'][0], cursor_point[1] - hotspot['pos'][1]) < hotspot['radius']:
                is_colliding = True

            if is_colliding:
                if hotspot['type'] == 'clue':
                    if hotspot['id'] not in self.collected_clue_ids:
                        self.collected_clue_ids.add(hotspot['id'])
                        self._add_feedback(f"Clue: {hotspot['text']}", 240, self.COLOR_GREEN_HINT)
                        # sfx: Clue found chime
                        return 10.0
                
                elif hotspot['type'] == 'witness':
                    self._add_feedback(hotspot['text'], 240, self.COLOR_TEXT)
                    # sfx: Dialogue blip
                    return 0.1

                elif hotspot['type'] == 'portal':
                    if hotspot['req'].issubset(self.collected_clue_ids):
                        target = hotspot['target']
                        reward = 0
                        if target not in self.visited_location_ids:
                            self.visited_location_ids.add(target)
                            reward = 5.0 # Big reward for finding new area
                        else:
                            reward = -0.1 # Small penalty for backtracking
                        self.current_location_id = target
                        self.cursor_pos = [self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2]
                        # sfx: Portal travel whoosh
                        return reward
                    else:
                        self._add_feedback(hotspot.get('locked_text', "This way is blocked."), 120, self.COLOR_RED_HINT)
                        # sfx: Locked/error sound
                        return -0.5
                
                # If any interaction happened, stop checking
                return 0.0
        return -0.2 # Penalty for interacting with nothing

    def _make_accusation(self):
        """Finalizes the accusation and ends the game."""
        selected_suspect = self.suspects[self.accusation_index]
        self.game_over = True
        reward = 0
        if selected_suspect == self.murderer:
            self.game_phase = 'RESULT_WIN'
            reward = 100.0
            # sfx: Success fanfare
        else:
            self.game_phase = 'RESULT_LOSE'
            reward = -100.0
            # sfx: Failure sound
        return reward

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "clues_found": len(self.collected_clue_ids),
            "game_phase": self.game_phase,
            "location": self.current_location_id
        }

    def _add_feedback(self, text, duration, color):
        """Adds a new text feedback message to the queue."""
        self.feedback_queue.insert(0, {'text': text, 'timer': duration, 'color': color})
        if len(self.feedback_queue) > 3:
            self.feedback_queue.pop()

    # --- Rendering Methods ---

    def _render_all(self):
        """Master render function."""
        self._render_background()
        self._render_game_elements()
        self._render_cursor()
        self._render_ui()
        if self.game_phase.startswith('RESULT'):
            self._render_result_screen()

    def _render_background(self):
        """Renders the shifting, dreamlike background."""
        bg_color = self.locations[self.current_location_id]['bg_color']
        self.screen.fill(bg_color)
        
        # Add a subtle, shifting overlay for a surreal effect
        for i in range(20):
            t = self.anim_tick * 0.01
            x = (self.SCREEN_WIDTH / 2) + math.sin(t + i) * 200 * math.cos(i)
            y = (self.SCREEN_HEIGHT / 2) + math.cos(t * 0.7 + i) * 150 * math.sin(i)
            radius = int(50 + 30 * math.sin(t * 0.5 + i * 0.5))
            color = (bg_color[0]+i, bg_color[1]+i, bg_color[2]+i, 4)
            pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), radius, color)

    def _render_game_elements(self):
        """Renders hotspots like clues, portals, and witnesses."""
        location_data = self.locations[self.current_location_id]
        cursor_point = tuple(self.cursor_pos)

        for hotspot in location_data['hotspots']:
            is_hover = False
            if 'rect' in hotspot and hotspot['rect'].collidepoint(cursor_point):
                is_hover = True
            elif 'pos' in hotspot and math.hypot(cursor_point[0] - hotspot['pos'][0], cursor_point[1] - hotspot['pos'][1]) < hotspot['radius']:
                is_hover = True

            if hotspot['type'] == 'clue':
                color = self.COLOR_CLUE_COLLECTED if hotspot['id'] in self.collected_clue_ids else self.COLOR_CLUE
                rect = hotspot['rect']
                pygame.draw.rect(self.screen, color, rect, border_radius=3)
                if hotspot['id'] not in self.collected_clue_ids:
                    # Shimmer effect
                    alpha = 100 + 80 * math.sin(self.anim_tick * 0.15)
                    glow_color = (*self.COLOR_CLUE, alpha)
                    glow_rect = rect.inflate(8, 8)
                    s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
                    pygame.draw.rect(s, glow_color, s.get_rect(), border_radius=6)
                    self.screen.blit(s, glow_rect.topleft)
                if is_hover:
                    pygame.draw.rect(self.screen, self.COLOR_TEXT, rect, 2, border_radius=3)

            elif hotspot['type'] == 'witness':
                pygame.draw.rect(self.screen, self.COLOR_WITNESS, hotspot['rect'], border_radius=5)
                if is_hover:
                    pygame.draw.rect(self.screen, self.COLOR_TEXT, hotspot['rect'], 2, border_radius=5)

            elif hotspot['type'] == 'portal':
                pos = hotspot['pos']
                is_unlocked = hotspot['req'].issubset(self.collected_clue_ids)
                radius = hotspot['radius'] + 5 * math.sin(self.anim_tick * 0.1)
                
                # Pulsating glow
                for i in range(3, 0, -1):
                    alpha = 50 if is_unlocked else 20
                    glow_color = (*self.COLOR_PORTAL, alpha / i)
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(radius + i * 5), glow_color)
                
                color = self.COLOR_PORTAL if is_unlocked else self.COLOR_WITNESS
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(radius), color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(radius), color)
                if is_hover:
                    pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(radius+3), self.COLOR_TEXT)
    
    def _render_cursor(self):
        """Renders the player's cursor with a glow."""
        pos = (int(self.cursor_pos[0]), int(self.cursor_pos[1]))
        # Glow effect
        s = pygame.Surface((32, 32), pygame.SRCALPHA)
        pygame.draw.circle(s, self.COLOR_CURSOR_GLOW, (16, 16), 16)
        self.screen.blit(s, (pos[0]-16, pos[1]-16))
        # Main cursor
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 6, self.COLOR_CURSOR)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 6, self.COLOR_CURSOR)

    def _render_ui(self):
        """Renders UI elements like score, steps, and feedback text."""
        # --- Top Bar Info ---
        score_text = self.font_main.render(f"Score: {self.score:.2f}", True, self.COLOR_TEXT)
        steps_text = self.font_main.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        clues_text = self.font_main.render(f"Clues: {len(self.collected_clue_ids)}/{len(self.all_clue_ids)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))
        self.screen.blit(clues_text, (self.SCREEN_WIDTH / 2 - clues_text.get_width() / 2, 10))

        # --- Location Name ---
        loc_name = self.locations[self.current_location_id]['name']
        loc_text = self.font_title.render(loc_name, True, self.COLOR_TEXT)
        self.screen.blit(loc_text, (self.SCREEN_WIDTH / 2 - loc_text.get_width() / 2, 40))

        # --- Feedback Text ---
        for i, f in enumerate(self.feedback_queue):
            alpha = max(0, min(255, f['timer'] * 4))
            self._draw_text_wrapped(f['text'], self.font_feedback, f['color'], 
                                    pygame.Rect(20, self.SCREEN_HEIGHT - 100 - (i*50), self.SCREEN_WIDTH - 40, 90),
                                    alpha)
        
        # --- Accusation UI ---
        if self.game_phase == 'ACCUSATION':
            self._render_accusation_ui()

    def _render_accusation_ui(self):
        """Renders the UI for making an accusation."""
        ui_rect = pygame.Rect(self.SCREEN_WIDTH // 2 - 200, self.SCREEN_HEIGHT // 2 - 100, 400, 200)
        s = pygame.Surface(ui_rect.size, pygame.SRCALPHA)
        s.fill(self.COLOR_ACCUSE_UI_BG)
        
        title_surf = self.font_title.render("Make Your Accusation", True, self.COLOR_TEXT)
        s.blit(title_surf, (ui_rect.width // 2 - title_surf.get_width() // 2, 15))
        
        help_surf = self.font_feedback.render("Use Shift to cycle, Space to accuse", True, self.COLOR_TEXT)
        s.blit(help_surf, (ui_rect.width // 2 - help_surf.get_width() // 2, 160))


        for i, suspect in enumerate(self.suspects):
            color = self.COLOR_ACCUSE_SELECTED if i == self.accusation_index else self.COLOR_TEXT
            suspect_surf = self.font_main.render(suspect, True, color)
            s.blit(suspect_surf, (ui_rect.width // 2 - suspect_surf.get_width() // 2, 60 + i * 30))

        self.screen.blit(s, ui_rect.topleft)

    def _render_result_screen(self):
        """Renders the final win/loss screen."""
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))

        if self.game_phase == 'RESULT_WIN':
            title_text = "Case Closed!"
            title_color = self.COLOR_GREEN_HINT
            msg_text = f"You correctly identified {self.murderer} as the killer."
        else:
            title_text = "Case Cold"
            title_color = self.COLOR_RED_HINT
            msg_text = f"You accused the wrong person. The killer was {self.murderer}."

        title_surf = self.font_title.render(title_text, True, title_color)
        msg_surf = self.font_main.render(msg_text, True, self.COLOR_TEXT)
        
        overlay.blit(title_surf, (self.SCREEN_WIDTH // 2 - title_surf.get_width() // 2, 150))
        overlay.blit(msg_surf, (self.SCREEN_WIDTH // 2 - msg_surf.get_width() // 2, 200))
        self.screen.blit(overlay, (0, 0))

    def _draw_text_wrapped(self, text, font, color, rect, alpha=255):
        """Draws text with word wrapping inside a given rect."""
        words = text.split(' ')
        lines = []
        current_line = ""
        for word in words:
            test_line = current_line + word + " "
            if font.size(test_line)[0] < rect.width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word + " "
        lines.append(current_line)
        
        y = rect.top
        line_height = font.get_linesize()
        
        # Background
        bg_rect = pygame.Rect(rect.left-5, rect.top-5, rect.width+10, len(lines)*line_height+10)
        s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        s.fill((*self.COLOR_TEXT_BG[:3], int(self.COLOR_TEXT_BG[3] * (alpha/255.0))))
        self.screen.blit(s, bg_rect.topleft)

        for line in lines:
            line_surf = font.render(line, True, (*color, alpha))
            self.screen.blit(line_surf, (rect.left, y))
            y += line_height

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("--- Running Implementation Validation ---")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2], f"Action space nvec is {self.action_space.nvec.tolist()}"
        print("✓ Action space validated.")
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3), f"Observation shape is {test_obs.shape}"
        assert test_obs.dtype == np.uint8, f"Observation dtype is {test_obs.dtype}"
        print("✓ Observation space validated.")
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        print("✓ reset() validated.")
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ step() validated.")
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Dreamscape Detective")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- Resetting Environment ---")
                env.reset()

        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Final Info: {info}")
            # In a real scenario, you'd wait for the reset key 'r'
            # For auto-play testing, you could just call env.reset() here.
            # running = False # uncomment to exit after one episode
            
        clock.tick(30) # Run at 30 FPS

    env.close()