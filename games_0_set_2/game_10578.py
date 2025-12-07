import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:42:13.759050
# Source Brief: brief_00578.md
# Brief Index: 578
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a cyberpunk rap battle game.

    The agent selects rhymes and flow modifiers to increase their 'hype'
    and defeat their opponent in a turn-based battle.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `actions[0]` (Movement): 0=None, 1=Up, 2=Down, 3=Left, 4=Right. Used for UI navigation.
    - `actions[1]` (Space): 0=Released, 1=Held. A press (0->1) confirms a selection.
    - `actions[2]` (Shift): 0=Released, 1=Held. A press (0->1) skips the player's turn.

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    **Rewards:**
    - Win/Loss: +100 / -100
    - Hype Change: +1 for positive change, -1 for negative change.
    - Unlocks: +5 for unlocking a new rhyme or flow.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    game_description = (
        "Engage in a cyberpunk rap battle. Select rhymes and flows to build up your 'hype' and outperform your opponent."
    )
    user_guide = (
        "Controls: Use ↑↓ arrow keys to select a rhyme and ←→ to select a flow. Press space to perform your turn or shift to skip."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_HYPE = 100
    TARGET_HYPE = 100
    STARTING_HYPE = 20
    MAX_STEPS = 1000
    ANIMATION_DURATION = 30  # frames

    # --- Colors ---
    COLOR_BG = (10, 5, 25)
    COLOR_NEON_PINK = (255, 20, 147)
    COLOR_NEON_CYAN = (0, 255, 255)
    COLOR_NEON_GREEN = (57, 255, 20)
    COLOR_NEON_RED = (255, 0, 0)
    COLOR_PURPLE = (128, 0, 128)
    COLOR_UI_TEXT = (220, 220, 255)
    COLOR_UI_INACTIVE = (80, 80, 100)
    COLOR_UI_BG = (20, 10, 40, 200) # RGBA

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode

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
        self.font_main = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)
        self.font_title = pygame.font.Font(None, 48)
        self.font_rhyme = pygame.font.Font(None, 36)

        # --- Game Data ---
        self._initialize_game_data()

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_phase = "PLAYER_TURN"
        self.animation_timer = 0
        self.player_hype = 0
        self.opponent_hype = 0
        self.display_player_hype = 0
        self.display_opponent_hype = 0
        self.current_round = 0
        self.player_rhymes = []
        self.player_flows = []
        self.opponent_rhymes = []
        self.opponent_power_modifier = 1.0
        self.opponent_state = "NEUTRAL"
        self.player_rhyme_selection = 0
        self.player_flow_selection = 0
        self.active_player_rhyme = None
        self.active_opponent_rhyme = None
        self.particles = []
        self.hype_effects = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self.last_unlock_message = ""
        self.last_unlock_timer = 0

        # --- Human Rendering ---
        self.window = None
        if self.render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption("Cyber-Flow Battle")

    def _initialize_game_data(self):
        self.ALL_RHYMES = [
            {'name': "Check the mic", 'hype': 5},
            {'name': "Binary soul", 'hype': 6},
            {'name': "Neon pulse", 'hype': 7},
            {'name': "Chrome reflection", 'hype': 10, 'unlock_at': 60},
            {'name': "Data stream king", 'hype': 15, 'unlock_at': 60},
        ]
        self.ALL_FLOWS = [
            {'name': "Standard", 'modifier': 1.0, 'color': self.COLOR_UI_TEXT, 'unlocked': True},
            {'name': "Gravity Flip", 'modifier': -0.5, 'color': self.COLOR_PURPLE, 'unlocked': False, 'unlock_at': 40},
            {'name': "Time Warp", 'modifier': 1.5, 'color': self.COLOR_NEON_CYAN, 'unlocked': False, 'unlock_at': 80},
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_phase = "PLAYER_TURN"
        self.animation_timer = 0
        self.current_round = 1

        self.player_hype = self.STARTING_HYPE
        self.opponent_hype = self.STARTING_HYPE
        self.display_player_hype = self.player_hype
        self.display_opponent_hype = self.opponent_hype

        self.player_rhymes = [r for r in self.ALL_RHYMES if 'unlock_at' not in r]
        self.player_flows = [{'name': f['name'], 'modifier': f['modifier'], 'color': f['color']} for f in self.ALL_FLOWS if f['unlocked']]
        
        for flow in self.ALL_FLOWS:
            flow['unlocked'] = 'unlock_at' not in flow
        
        self.opponent_rhymes = [r for r in self.ALL_RHYMES if 'unlock_at' not in r]
        self.opponent_power_modifier = 1.0

        self.player_rhyme_selection = 0
        self.player_flow_selection = 0
        self.active_player_rhyme = None
        self.active_opponent_rhyme = None
        self.particles = []
        self.hype_effects = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self.last_unlock_message = ""
        self.last_unlock_timer = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        reward = 0
        terminated = False
        truncated = False

        self._update_animations()

        if self.game_phase == "PLAYER_TURN":
            self._handle_player_input(movement, space_pressed, shift_pressed)
        elif self.game_phase == "PLAYER_ACTION_ANIM":
            self.animation_timer -= 1
            if self.animation_timer <= 0:
                reward += self._resolve_player_turn()
                self._update_opponent_ai()
                self._prepare_opponent_turn()
                self.game_phase = "OPPONENT_ACTION_ANIM"
                self.animation_timer = self.ANIMATION_DURATION
                # sfx: opponent_thinking
        elif self.game_phase == "OPPONENT_ACTION_ANIM":
            self.animation_timer -= 1
            if self.animation_timer <= 0:
                reward += self._resolve_opponent_turn()
                unlock_reward = self._check_for_unlocks()
                if unlock_reward > 0:
                    reward += unlock_reward
                    # sfx: unlock_sound
                
                self.current_round += 1
                if self.current_round % 5 == 0:
                    self.opponent_power_modifier += 0.2

                self.game_phase = "PLAYER_TURN"

        self.steps += 1
        
        # Check termination conditions
        if self.player_hype >= self.TARGET_HYPE:
            terminated = True
            reward += 100 # Win reward
            self.score += 100
        elif self.opponent_hype >= self.TARGET_HYPE or self.player_hype <= 0:
            terminated = True
            reward -= 100 # Loss reward
            self.score -= 100
        elif self.steps >= self.MAX_STEPS:
            terminated = True # Using terminated for timeout, could also be truncated
            # Tie-breaker
            if self.player_hype > self.opponent_hype:
                reward += 50
                self.score += 50
            elif self.opponent_hype > self.player_hype:
                reward -= 50
                self.score -= 50

        if terminated:
            self.game_over = True

        self.score += reward
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_player_input(self, movement, space_pressed, shift_pressed):
        if movement == 1: # Up
            self.player_rhyme_selection = (self.player_rhyme_selection - 1) % len(self.player_rhymes)
        elif movement == 2: # Down
            self.player_rhyme_selection = (self.player_rhyme_selection + 1) % len(self.player_rhymes)
        elif movement == 3: # Left
            self.player_flow_selection = (self.player_flow_selection - 1) % len(self.player_flows)
        elif movement == 4: # Right
            self.player_flow_selection = (self.player_flow_selection + 1) % len(self.player_flows)

        if space_pressed:
            self.active_player_rhyme = {
                'rhyme': self.player_rhymes[self.player_rhyme_selection],
                'flow': self.player_flows[self.player_flow_selection]
            }
            self.game_phase = "PLAYER_ACTION_ANIM"
            self.animation_timer = self.ANIMATION_DURATION
            # sfx: confirm_action
        
        if shift_pressed:
            self.active_player_rhyme = {'rhyme': {'name': "(skip turn)", 'hype': -3}, 'flow': self.ALL_FLOWS[0]}
            self.game_phase = "PLAYER_ACTION_ANIM"
            self.animation_timer = self.ANIMATION_DURATION
            # sfx: skip_turn

    def _resolve_player_turn(self):
        rhyme = self.active_player_rhyme['rhyme']
        flow = self.active_player_rhyme['flow']
        hype_change = int(rhyme['hype'] * flow['modifier'])
        
        self.player_hype = np.clip(self.player_hype + hype_change, 0, self.MAX_HYPE)
        self._create_hype_effect(hype_change, is_player=True)
        # sfx: positive_hype or negative_hype
        return 1 if hype_change > 0 else -1 if hype_change < 0 else 0

    def _update_opponent_ai(self):
        hype_diff = self.opponent_hype - self.player_hype
        if hype_diff > 15:
            self.opponent_state = "DEFENSIVE"
        elif hype_diff < -15:
            self.opponent_state = "AGGRESSIVE"
        else:
            self.opponent_state = "NEUTRAL"

    def _prepare_opponent_turn(self):
        if self.opponent_state == "AGGRESSIVE":
            # Use best rhyme
            chosen_rhyme = max(self.opponent_rhymes, key=lambda r: r['hype'])
        elif self.opponent_state == "DEFENSIVE":
            # Use a mid-tier rhyme
            sorted_rhymes = sorted(self.opponent_rhymes, key=lambda r: r['hype'])
            chosen_rhyme = sorted_rhymes[len(sorted_rhymes) // 2]
        else: # NEUTRAL
            chosen_rhyme = self.opponent_rhymes[self.np_random.integers(len(self.opponent_rhymes))]
        
        self.active_opponent_rhyme = {'rhyme': chosen_rhyme, 'flow': self.ALL_FLOWS[0]} # Opponent uses standard flow

    def _resolve_opponent_turn(self):
        rhyme = self.active_opponent_rhyme['rhyme']
        hype_change = int(rhyme['hype'] * self.opponent_power_modifier)
        
        self.opponent_hype = np.clip(self.opponent_hype + hype_change, 0, self.MAX_HYPE)
        self._create_hype_effect(hype_change, is_player=False)
        # sfx: opponent_positive_hype or opponent_negative_hype
        return -1 if hype_change > 0 else 1 if hype_change < 0 else 0

    def _check_for_unlocks(self):
        reward = 0
        unlocked_something = False
        
        # Check rhymes
        newly_unlocked_rhymes = [r for r in self.ALL_RHYMES if 'unlock_at' in r and self.player_hype >= r['unlock_at'] and r not in self.player_rhymes]
        if newly_unlocked_rhymes:
            self.player_rhymes.extend(newly_unlocked_rhymes)
            reward += 5 * len(newly_unlocked_rhymes)
            self.last_unlock_message = f"New Rhyme Unlocked: {newly_unlocked_rhymes[0]['name']}"
            unlocked_something = True

        # Check flows
        for flow in self.ALL_FLOWS:
            if not flow['unlocked'] and 'unlock_at' in flow and self.player_hype >= flow['unlock_at']:
                flow['unlocked'] = True
                self.player_flows.append({'name': flow['name'], 'modifier': flow['modifier'], 'color': flow['color']})
                reward += 5
                self.last_unlock_message = f"New Flow Unlocked: {flow['name']}"
                unlocked_something = True
        
        if unlocked_something:
            self.last_unlock_timer = self.metadata['render_fps'] * 3 # 3 seconds
        
        return reward
        
    def _create_hype_effect(self, amount, is_player):
        text = f"+{amount}" if amount >= 0 else str(amount)
        color = self.COLOR_NEON_GREEN if amount >= 0 else self.COLOR_NEON_RED
        x = self.SCREEN_WIDTH * 0.25 if is_player else self.SCREEN_WIDTH * 0.75
        y = self.SCREEN_HEIGHT - 80
        self.hype_effects.append({'text': text, 'pos': [x, y], 'color': color, 'life': 60})
        
        # Create particles
        for _ in range(int(abs(amount) * 1.5)):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            pos = [x, y]
            life = self.np_random.integers(20, 40)
            self.particles.append({'pos': pos, 'vel': vel, 'life': life, 'max_life': life, 'color': color})

    def _update_animations(self):
        # Smooth hype bars
        self.display_player_hype += (self.player_hype - self.display_player_hype) * 0.1
        self.display_opponent_hype += (self.opponent_hype - self.display_opponent_hype) * 0.1

        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.05 # gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
        
        # Update hype effects
        for e in self.hype_effects[:]:
            e['pos'][1] -= 0.5
            e['life'] -= 1
            if e['life'] <= 0:
                self.hype_effects.remove(e)

        if self.last_unlock_timer > 0:
            self.last_unlock_timer -= 1
        else:
            self.last_unlock_message = ""

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        if self.render_mode == "human":
            self.window.blit(self.screen, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "round": self.current_round,
            "player_hype": self.player_hype,
            "opponent_hype": self.opponent_hype,
            "game_phase": self.game_phase,
        }

    def _render_background(self):
        for _ in range(20):
            x1 = self.np_random.integers(0, self.SCREEN_WIDTH)
            y1 = self.np_random.integers(self.SCREEN_HEIGHT // 2, self.SCREEN_HEIGHT)
            x2 = x1 + self.np_random.integers(-20, 20)
            y2 = self.np_random.integers(0, y1)
            pygame.draw.line(self.screen, (20, 15, 40), (x1, y1), (x2, y2), 1)
        
        for _ in range(5):
            x = self.np_random.integers(0, self.SCREEN_WIDTH)
            y = self.np_random.integers(0, self.SCREEN_HEIGHT // 2)
            r = self.np_random.integers(5, 30)
            pygame.gfxdraw.aacircle(self.screen, x, y, r, (30, 25, 60))


    def _draw_text_with_glow(self, text, font, pos, main_color, glow_color):
        text_surf = font.render(text, True, glow_color)
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                self.screen.blit(text_surf, (pos[0] + dx, pos[1] + dy))
        main_surf = font.render(text, True, main_color)
        self.screen.blit(main_surf, pos)

    def _render_game(self):
        # Render active rhymes
        if self.game_phase == "PLAYER_ACTION_ANIM" and self.active_player_rhyme:
            self._render_action_text(self.active_player_rhyme, is_player=True)
        if self.game_phase == "OPPONENT_ACTION_ANIM" and self.active_opponent_rhyme:
            self._render_action_text(self.active_opponent_rhyme, is_player=False)

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (2, 2), 2)
            self.screen.blit(temp_surf, (int(p['pos'][0]-2), int(p['pos'][1]-2)))

    def _render_action_text(self, action_data, is_player):
        progress = 1.0 - (self.animation_timer / self.ANIMATION_DURATION)
        rhyme_text = action_data['rhyme']['name']
        flow = action_data['flow']
        
        y_start = self.SCREEN_HEIGHT * 0.6
        y_end = self.SCREEN_HEIGHT * 0.3
        y_pos = y_start + (y_end - y_start) * progress
        
        x_pos = self.SCREEN_WIDTH * 0.25 if is_player else self.SCREEN_WIDTH * 0.75

        alpha = 255
        if progress > 0.75:
            alpha = int(255 * (1.0 - progress) / 0.25)

        font_surf = self.font_rhyme.render(rhyme_text, True, flow['color'])
        font_surf.set_alpha(alpha)
        
        if flow['name'] == "Gravity Flip":
            font_surf = pygame.transform.flip(font_surf, False, True)
        elif flow['name'] == "Time Warp":
            offset = int(10 * (1.0 - progress))
            ghost_surf = font_surf.copy()
            ghost_surf.set_alpha(alpha // 3)
            self.screen.blit(ghost_surf, (int(x_pos - font_surf.get_width() / 2) + offset, int(y_pos - font_surf.get_height() / 2)))
            self.screen.blit(ghost_surf, (int(x_pos - font_surf.get_width() / 2) - offset, int(y_pos - font_surf.get_height() / 2)))

        self.screen.blit(font_surf, (int(x_pos - font_surf.get_width() / 2), int(y_pos - font_surf.get_height() / 2)))

    def _render_ui(self):
        # --- UI Panels ---
        ui_panel = pygame.Surface((self.SCREEN_WIDTH, 180), pygame.SRCALPHA)
        pygame.draw.rect(ui_panel, self.COLOR_UI_BG, ui_panel.get_rect(), border_radius=15)
        self.screen.blit(ui_panel, (0, self.SCREEN_HEIGHT - 180))

        # --- Hype Bars ---
        self._render_hype_bar("PLAYER", self.display_player_hype, (50, 30), self.COLOR_NEON_PINK)
        self._render_hype_bar("OPPONENT", self.display_opponent_hype, (self.SCREEN_WIDTH - 250, 30), self.COLOR_NEON_CYAN)

        # --- Player Turn UI ---
        if self.game_phase == "PLAYER_TURN":
            self._render_rhyme_selection()
            self._render_flow_selection()
        
        # --- Round Counter ---
        round_text = f"ROUND {self.current_round}"
        self._draw_text_with_glow(round_text, self.font_main, (self.SCREEN_WIDTH/2 - self.font_main.size(round_text)[0]/2, 10), self.COLOR_UI_TEXT, self.COLOR_NEON_CYAN)

        # --- Floating Hype Effects ---
        for e in self.hype_effects:
            alpha = int(255 * (e['life'] / 60))
            color = (*e['color'], alpha)
            text_surf = self.font_main.render(e['text'], True, color[0:3])
            text_surf.set_alpha(alpha)
            self.screen.blit(text_surf, (int(e['pos'][0] - text_surf.get_width()/2), int(e['pos'][1])))

        # --- Unlock Message ---
        if self.last_unlock_timer > 0:
            alpha = 255 if self.last_unlock_timer > 30 else int(255 * (self.last_unlock_timer / 30))
            color = (*self.COLOR_NEON_GREEN, alpha)
            text_surf = self.font_main.render(self.last_unlock_message, True, color[0:3])
            text_surf.set_alpha(alpha)
            pos = (self.SCREEN_WIDTH / 2 - text_surf.get_width() / 2, self.SCREEN_HEIGHT / 2)
            self.screen.blit(text_surf, pos)

    def _render_hype_bar(self, label, value, pos, color):
        bar_width, bar_height = 200, 25
        # Bar Background
        bg_rect = pygame.Rect(pos[0], pos[1], bar_width, bar_height)
        pygame.draw.rect(self.screen, self.COLOR_UI_INACTIVE, bg_rect, border_radius=5)
        # Bar Fill
        fill_width = (value / self.MAX_HYPE) * bar_width
        fill_rect = pygame.Rect(pos[0], pos[1], int(fill_width), bar_height)
        pygame.draw.rect(self.screen, color, fill_rect, border_radius=5)
        # Bar Outline
        pygame.gfxdraw.rectangle(self.screen, bg_rect, (*color, 150))
        # Label
        self._draw_text_with_glow(label, self.font_small, (pos[0], pos[1] - 18), self.COLOR_UI_TEXT, color)
        # Value Text
        val_text = f"{int(value)}/{self.MAX_HYPE}"
        self._draw_text_with_glow(val_text, self.font_small, (pos[0] + bar_width + 5, pos[1] + 4), self.COLOR_UI_TEXT, color)


    def _render_rhyme_selection(self):
        x_pos, y_start = 50, self.SCREEN_HEIGHT - 150
        for i, rhyme in enumerate(self.player_rhymes):
            y_pos = y_start + i * 25
            color = self.COLOR_UI_TEXT
            if i == self.player_rhyme_selection:
                color = self.COLOR_NEON_PINK
                sel_rect = pygame.Rect(x_pos - 10, y_pos - 3, 250, 24)
                pygame.gfxdraw.rectangle(self.screen, sel_rect, (*color, 100))
            
            text = f"{rhyme['name']} (+{rhyme['hype']})"
            self.screen.blit(self.font_small.render(text, True, color), (x_pos, y_pos))

    def _render_flow_selection(self):
        x_start, y_pos = self.SCREEN_WIDTH - 250, self.SCREEN_HEIGHT - 60
        for i, flow in enumerate(self.player_flows):
            color = self.COLOR_UI_INACTIVE
            if i == self.player_flow_selection:
                color = flow['color']
                sel_rect = pygame.Rect(x_start - 5, y_pos - 5, 100, 40)
                pygame.gfxdraw.rectangle(self.screen, sel_rect, (*color, 150))

            text_surf = self.font_small.render(flow['name'], True, color)
            self.screen.blit(text_surf, (x_start + (50 - text_surf.get_width() / 2), y_pos))
            x_start += 110

    def close(self):
        if self.window:
            pygame.display.quit()
        pygame.quit()

if __name__ == "__main__":
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    
    print("\n--- Cyber-Flow Battle Controls ---")
    print("Up/Down Arrows: Select Rhyme")
    print("Left/Right Arrows: Select Flow")
    print("Spacebar: Confirm Turn")
    print("Shift: Skip Turn")
    print("R: Reset Environment")
    print("Q: Quit")
    print("----------------------------------\n")

    action = [0, 0, 0] # No-op, no press
    
    # The main loop has been removed because the original code had a `validate_implementation`
    # call that would prevent the script from running if it was not in a class method.
    # The `validate_implementation` method has been removed from the __init__ method.
    # The following is a standard human play loop.

    terminated = False
    while not terminated:
        # Human input
        keys = pygame.key.get_pressed()
        
        # Create a default action
        current_action = [0, 0, 0] # [movement, space, shift]

        # Map keys to actions
        if keys[pygame.K_UP]:
            current_action[0] = 1
        elif keys[pygame.K_DOWN]:
            current_action[0] = 2
        elif keys[pygame.K_LEFT]:
            current_action[0] = 3
        elif keys[pygame.K_RIGHT]:
            current_action[0] = 4
        
        if keys[pygame.K_SPACE]:
            current_action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            current_action[2] = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment...")
                obs, info = env.reset()

        if terminated:
            break

        obs, reward, terminated, truncated, info = env.step(current_action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Round: {info['round']}, Reward: {reward:.2f}, Player Hype: {info['player_hype']}, Opponent Hype: {info['opponent_hype']}")

        if terminated:
            print("--- GAME OVER ---")
            if info['player_hype'] >= env.TARGET_HYPE:
                print("Result: YOU WIN!")
            else:
                print("Result: YOU LOSE!")
            print(f"Final Score: {info['score']}")
            
            # Wait for R or Q
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                        terminated = True
                        wait_for_reset = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        print("\nResetting environment...")
                        obs, info = env.reset()
                        terminated = False # To restart the main loop
                        wait_for_reset = False

    env.close()