import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:26:25.610629
# Source Brief: brief_00399.md
# Brief Index: 399
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Stabilize volatile quantum anomalies by strategically deploying different types of particles. "
        "Create chain reactions and unlock new tools to solve each puzzle."
    )
    user_guide = (
        "Controls: Use ↑/↓ to select a particle card. Press space to enter targeting mode. "
        "Use arrow keys to move the target, space to deploy, and shift to cancel."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 18, 10
    GRID_LEFT, GRID_TOP = 40, 40
    CELL_SIZE = 30
    MAX_STEPS = 500

    # Colors
    COLOR_BG = (15, 18, 26)
    COLOR_GRID = (30, 35, 50)
    COLOR_TEXT = (220, 230, 255)
    COLOR_TEXT_DIM = (100, 110, 130)
    COLOR_CURSOR = (0, 255, 255)
    COLOR_UNSTABLE = (255, 80, 80)
    COLOR_STABLE = (80, 180, 255)

    PARTICLE_COLORS = {
        "EXPLOSIVE": (255, 200, 0),
        "STABILIZING": (0, 255, 150),
        "CHAIN": (200, 100, 255),
        "REMOVER": (240, 240, 240),
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        
        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.anomalies = []
        self.grid_particles = []
        self.visual_effects = []
        self.deck = []
        self.hand = []
        self.interaction_state = "SELECTING_CARD"
        self.selected_card_idx = 0
        self.target_cursor_pos = [0, 0]
        self.prev_space_held = False
        self.prev_shift_held = False
        self.anomalies_stabilized_count = 0
        self.total_reward_this_step = 0
        
        # Persistent state (across resets within one env instance)
        self.unlocked_particle_types = ["EXPLOSIVE", "STABILIZING"]

        # self.reset() # reset is called by the wrapper
        # self.validate_implementation() # this is for debugging, not needed in final version

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.anomalies = []
        self.grid_particles = []
        self.visual_effects = []
        self.anomalies_stabilized_count = 0
        
        self._generate_level(num_anomalies=self.np_random.integers(3, 5))

        self.deck = []
        for p_type in self.unlocked_particle_types:
            self.deck.extend([p_type] * 10)
        
        # In case this is not the first run, ensure unlocked particles are shuffled in
        if "CHAIN" in self.unlocked_particle_types and "CHAIN" not in self.deck:
             self.deck.extend(["CHAIN"] * 10)
        if "REMOVER" in self.unlocked_particle_types and "REMOVER" not in self.deck:
             self.deck.extend(["REMOVER"] * 10)
        
        self.np_random.shuffle(self.deck)

        self.hand = [self._draw_card() for _ in range(5)]
        self.hand = [card for card in self.hand if card is not None]

        self.interaction_state = "SELECTING_CARD"
        self.selected_card_idx = 0
        self.target_cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        
        self.prev_space_held = True # Prevent action on first frame
        self.prev_shift_held = True

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.total_reward_this_step = 0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        self._handle_input(movement, space_pressed, shift_pressed)
        self._update_visual_effects()
        
        reward = self.total_reward_this_step
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            self.game_over = True
            all_stabilized = all(a.is_stable() for a in self.anomalies)
            if all_stabilized:
                reward += 50 # Win bonus
                self.visual_effects.append(WinLossEffect("LEVEL STABILIZED", (0, 255, 150)))
            elif not self.hand:
                reward -= 10 # Lose penalty
                self.visual_effects.append(WinLossEffect("DECK EMPTY", (255, 80, 80)))
            else: # Max steps
                self.visual_effects.append(WinLossEffect("TIMEOUT", (255, 200, 0)))


        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_pressed, shift_pressed):
        if self.interaction_state == "SELECTING_CARD":
            if movement in [1, 2]: # Up/Down
                # # Sound: UI_NAVIGATE
                if movement == 1: self.selected_card_idx -= 1
                if movement == 2: self.selected_card_idx += 1
                self.selected_card_idx = np.clip(self.selected_card_idx, 0, len(self.hand) - 1 if self.hand else 0)
            
            if space_pressed and self.hand:
                # # Sound: UI_CONFIRM
                self.interaction_state = "SELECTING_TARGET"

        elif self.interaction_state == "SELECTING_TARGET":
            if movement == 1: self.target_cursor_pos[1] -= 1 # Up
            if movement == 2: self.target_cursor_pos[1] += 1 # Down
            if movement == 3: self.target_cursor_pos[0] -= 1 # Left
            if movement == 4: self.target_cursor_pos[0] += 1 # Right
            
            self.target_cursor_pos[0] = np.clip(self.target_cursor_pos[0], 0, self.GRID_COLS - 1)
            self.target_cursor_pos[1] = np.clip(self.target_cursor_pos[1], 0, self.GRID_ROWS - 1)

            if shift_pressed:
                # # Sound: UI_CANCEL
                self.interaction_state = "SELECTING_CARD"
            
            if space_pressed:
                # # Sound: PARTICLE_DEPLOY
                self._deploy_particle()
                self.interaction_state = "SELECTING_CARD"

    def _deploy_particle(self):
        card_type = self.hand.pop(self.selected_card_idx)
        new_particle = Particle(card_type, tuple(self.target_cursor_pos))
        
        # Don't add REMOVER to grid, it's an instant effect
        if card_type != "REMOVER":
            self.grid_particles.append(new_particle)
        
        self._simulate_reactions(new_particle)
        
        new_card = self._draw_card()
        if new_card:
            self.hand.append(new_card)
        
        self.selected_card_idx = np.clip(self.selected_card_idx, 0, len(self.hand) - 1 if self.hand else 0)

    def _simulate_reactions(self, initiator):
        initial_stabilities = {i: a.stability for i, a in enumerate(self.anomalies)}
        
        q = deque([initiator])
        processed_chain = {initiator.pos}

        # --- REMOVER ---
        if initiator.p_type == "REMOVER":
            removed_particles = []
            for p in self.grid_particles[:]:
                if math.dist(p.pos, initiator.pos) <= 1.5:
                    removed_particles.append(p)
                    self.grid_particles.remove(p)
            if removed_particles:
                # # Sound: REMOVER_ACTIVATE
                self.visual_effects.append(ImplosionEffect(initiator.pos, self.PARTICLE_COLORS[initiator.p_type], 20))

        # --- EXPLOSIVE ---
        elif initiator.p_type == "EXPLOSIVE":
            # # Sound: EXPLOSION
            radius = 2.5
            self.visual_effects.append(ExplosionEffect(initiator.pos, self.PARTICLE_COLORS[initiator.p_type], radius * self.CELL_SIZE))
            for anom in self.anomalies:
                if math.dist(initiator.pos, anom.pos) < radius:
                    anom.change_stability(-25)

        # --- STABILIZING ---
        elif initiator.p_type == "STABILIZING":
            # # Sound: STABILIZE_PULSE
            radius = 2.5
            self.visual_effects.append(PulseEffect(initiator.pos, self.PARTICLE_COLORS[initiator.p_type], radius * self.CELL_SIZE))
            for anom in self.anomalies:
                if math.dist(initiator.pos, anom.pos) < radius:
                    anom.change_stability(25)

        # --- CHAIN ---
        elif initiator.p_type == "CHAIN":
            # # Sound: CHAIN_START
            chain_links = [initiator.pos]
            while q:
                current_p = q.popleft()
                # # Sound: CHAIN_LINK
                
                # Effect on adjacent anomalies
                for anom in self.anomalies:
                    if math.dist(current_p.pos, anom.pos) < 1.5:
                        anom.change_stability(10)

                # Find next links in chain
                for p in self.grid_particles:
                    if p.p_type == "CHAIN" and p.pos not in processed_chain:
                        dx, dy = abs(p.pos[0] - current_p.pos[0]), abs(p.pos[1] - current_p.pos[1])
                        if (dx == 1 and dy == 0) or (dx == 0 and dy == 1):
                            q.append(p)
                            processed_chain.add(p.pos)
                            chain_links.append(p.pos)
            
            if len(chain_links) > 1:
                self.visual_effects.append(ChainLinkEffect(chain_links, self.PARTICLE_COLORS["CHAIN"]))

        # Calculate rewards and handle stabilization
        for i, anom in enumerate(self.anomalies):
            stability_change = anom.stability - initial_stabilities[i]
            if stability_change != 0:
                # Reward for stability change
                self.total_reward_this_step += stability_change
                self.score += stability_change
                self.visual_effects.append(TextPopupEffect(anom.pos, f"{stability_change:+.0f}%", self.COLOR_STABLE if stability_change > 0 else self.COLOR_UNSTABLE))

            if anom.stability >= 100 and initial_stabilities[i] < 100:
                # # Sound: ANOMALY_STABILIZED
                self.total_reward_this_step += 5
                self.score += 5
                self.anomalies_stabilized_count += 1
                self.visual_effects.append(TextPopupEffect(anom.pos, "STABILIZED!", self.COLOR_STABLE, size='large'))
                self._check_unlocks()

    def _check_unlocks(self):
        if "CHAIN" not in self.unlocked_particle_types and self.anomalies_stabilized_count >= 2:
            self.unlocked_particle_types.append("CHAIN")
            self._add_to_deck("CHAIN")
            self.visual_effects.append(UnlockEffect("CHAIN PARTICLE UNLOCKED"))
        if "REMOVER" not in self.unlocked_particle_types and self.anomalies_stabilized_count >= 4:
            self.unlocked_particle_types.append("REMOVER")
            self._add_to_deck("REMOVER")
            self.visual_effects.append(UnlockEffect("REMOVER PARTICLE UNLOCKED"))
            
    def _add_to_deck(self, p_type):
        # # Sound: UNLOCK
        new_cards = [p_type] * 10
        self.deck.extend(new_cards)
        self.np_random.shuffle(self.deck)

    def _generate_level(self, num_anomalies):
        occupied_pos = set()
        for _ in range(num_anomalies):
            pos = (self.np_random.integers(0, self.GRID_COLS), self.np_random.integers(0, self.GRID_ROWS))
            while pos in occupied_pos:
                pos = (self.np_random.integers(0, self.GRID_COLS), self.np_random.integers(0, self.GRID_ROWS))
            occupied_pos.add(pos)
            self.anomalies.append(Anomaly(pos, self.np_random.integers(40, 60)))

    def _draw_card(self):
        if not self.deck:
            return None
        return self.deck.pop(0)

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            return True
        if not self.hand and self.interaction_state == "SELECTING_CARD":
            return True
        if all(a.is_stable() for a in self.anomalies):
            return True
        return False

    def _update_visual_effects(self):
        for effect in self.visual_effects[:]:
            effect.update()
            if not effect.is_alive():
                self.visual_effects.remove(effect)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_anomalies()
        self._render_particles()
        if not self.game_over:
            self._render_cursor()
        self._render_visual_effects()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "anomalies_stabilized": self.anomalies_stabilized_count,
            "cards_in_deck": len(self.deck),
        }

    def _grid_to_pixel(self, grid_pos):
        x = self.GRID_LEFT + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2
        y = self.GRID_TOP + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2
        return int(x), int(y)

    # --- RENDER METHODS ---

    def _render_grid(self):
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_TOP + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_LEFT, y), (self.GRID_LEFT + self.GRID_COLS * self.CELL_SIZE, y))
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_LEFT + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_TOP), (x, self.GRID_TOP + self.GRID_ROWS * self.CELL_SIZE))

    def _render_anomalies(self):
        for anom in self.anomalies:
            anom.draw(self.screen, self)

    def _render_particles(self):
        for particle in self.grid_particles:
            particle.draw(self.screen, self)

    def _render_cursor(self):
        if self.interaction_state == "SELECTING_TARGET":
            px, py = self._grid_to_pixel(self.target_cursor_pos)
            size = self.CELL_SIZE // 2
            glow = int(abs(math.sin(pygame.time.get_ticks() * 0.005)) * 5)
            
            # Glow effect
            pygame.gfxdraw.aacircle(self.screen, px, py, size + glow, (*self.COLOR_CURSOR, 50))
            pygame.gfxdraw.filled_circle(self.screen, px, py, size + glow, (*self.COLOR_CURSOR, 50))
            
            # Crosshair
            pygame.draw.line(self.screen, self.COLOR_CURSOR, (px - size, py), (px + size, py), 2)
            pygame.draw.line(self.screen, self.COLOR_CURSOR, (px, py - size), (px, py + size), 2)

    def _render_visual_effects(self):
        for effect in self.visual_effects:
            effect.draw(self.screen, self)

    def _render_ui(self):
        # Score and info
        score_text = self.font_medium.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        deck_text = self.font_medium.render(f"DECK: {len(self.deck)}", True, self.COLOR_TEXT)
        self.screen.blit(deck_text, (self.SCREEN_WIDTH - deck_text.get_width() - 10, 10))

        # Hand
        card_width, card_height = 100, 60
        hand_y = self.SCREEN_HEIGHT - card_height - 10
        total_hand_width = len(self.hand) * (card_width + 10) - 10
        hand_x_start = (self.SCREEN_WIDTH - total_hand_width) / 2

        for i, card_type in enumerate(self.hand):
            card_x = hand_x_start + i * (card_width + 10)
            rect = pygame.Rect(card_x, hand_y, card_width, card_height)
            
            is_selected = (i == self.selected_card_idx and self.interaction_state != "GAME_OVER")
            
            border_color = self.COLOR_CURSOR if is_selected else self.PARTICLE_COLORS[card_type]
            bg_color = (*self.PARTICLE_COLORS[card_type], 40) if is_selected else (25, 30, 45)
            
            pygame.gfxdraw.box(self.screen, rect, bg_color)
            pygame.draw.rect(self.screen, border_color, rect, 2, border_radius=5)
            
            card_name = self.font_medium.render(card_type, True, self.COLOR_TEXT)
            self.screen.blit(card_name, (rect.centerx - card_name.get_width() / 2, rect.centery - card_name.get_height() / 2))

        # Prompt
        if not self.game_over:
            prompt_text_str = ""
            if self.interaction_state == "SELECTING_CARD":
                prompt_text_str = "▲/▼: Select Card | [SPACE]: Choose Target"
            elif self.interaction_state == "SELECTING_TARGET":
                prompt_text_str = "ARROWS: Move Target | [SPACE]: Deploy | [SHIFT]: Cancel"
            prompt_text = self.font_small.render(prompt_text_str, True, self.COLOR_TEXT_DIM)
            self.screen.blit(prompt_text, (self.SCREEN_WIDTH/2 - prompt_text.get_width()/2, self.SCREEN_HEIGHT - 80))

# --- HELPER CLASSES ---

class Anomaly:
    def __init__(self, pos, stability):
        self.pos = pos
        self.stability = np.clip(stability, 0, 100)
        self.radius = GameEnv.CELL_SIZE * 0.4
        self.pulse = 0

    def is_stable(self):
        return self.stability >= 100

    def change_stability(self, amount):
        if not self.is_stable():
            self.stability = np.clip(self.stability + amount, 0, 100)
    
    def draw(self, surface, env):
        px, py = env._grid_to_pixel(self.pos)
        self.pulse = (self.pulse + 0.05) % (2 * math.pi)
        
        color_lerp = self.stability / 100.0
        color = (
            int(GameEnv.COLOR_UNSTABLE[0] * (1 - color_lerp) + GameEnv.COLOR_STABLE[0] * color_lerp),
            int(GameEnv.COLOR_UNSTABLE[1] * (1 - color_lerp) + GameEnv.COLOR_STABLE[1] * color_lerp),
            int(GameEnv.COLOR_UNSTABLE[2] * (1 - color_lerp) + GameEnv.COLOR_STABLE[2] * color_lerp),
        )
        
        current_radius = self.radius
        if not self.is_stable():
             current_radius += math.sin(self.pulse) * 3
        
        # Glow
        glow_radius = int(current_radius + 5 + math.sin(self.pulse) * 3)
        glow_alpha = int(50 + 30 * math.sin(self.pulse))
        pygame.gfxdraw.filled_circle(surface, px, py, glow_radius, (*color, glow_alpha))
        
        # Main circle
        pygame.gfxdraw.aacircle(surface, px, py, int(current_radius), color)
        pygame.gfxdraw.filled_circle(surface, px, py, int(current_radius), color)
        
        # Stability text
        if not self.is_stable():
            stab_text = env.font_small.render(f"{int(self.stability)}%", True, GameEnv.COLOR_TEXT)
            surface.blit(stab_text, (px - stab_text.get_width()/2, py - stab_text.get_height()/2))

class Particle:
    def __init__(self, p_type, pos):
        self.p_type = p_type
        self.pos = pos
        self.color = GameEnv.PARTICLE_COLORS[p_type]
        self.radius = GameEnv.CELL_SIZE * 0.2
        self.spawn_time = pygame.time.get_ticks()

    def draw(self, surface, env):
        px, py = env._grid_to_pixel(self.pos)
        age = (pygame.time.get_ticks() - self.spawn_time) / 1000.0
        
        # Pulsating spawn effect
        size_mult = min(1.0, age * 2.0)
        current_radius = self.radius * size_mult
        
        pygame.gfxdraw.aacircle(surface, px, py, int(current_radius), self.color)
        pygame.gfxdraw.filled_circle(surface, px, py, int(current_radius), self.color)
        
        # Inner dot
        pygame.gfxdraw.filled_circle(surface, px, py, int(current_radius/2), (255,255,255))

class Effect:
    def __init__(self, pos, lifetime):
        self.pos = pos
        self.lifetime = lifetime
        self.max_lifetime = lifetime

    def update(self):
        self.lifetime -= 1

    def is_alive(self):
        return self.lifetime > 0

    def draw(self, surface, env):
        pass

class ExplosionEffect(Effect):
    def __init__(self, pos, color, max_radius):
        super().__init__(pos, 30)
        self.color = color
        self.max_radius = max_radius

    def draw(self, surface, env):
        progress = 1.0 - (self.lifetime / self.max_lifetime)
        current_radius = int(self.max_radius * progress)
        alpha = int(200 * (1.0 - progress))
        px, py = env._grid_to_pixel(self.pos)
        
        if alpha > 0 and current_radius > 0:
            pygame.gfxdraw.aacircle(surface, px, py, current_radius, (*self.color, alpha))
            pygame.gfxdraw.aacircle(surface, px, py, current_radius-1, (*self.color, alpha))

class ImplosionEffect(ExplosionEffect):
     def draw(self, surface, env):
        progress = self.lifetime / self.max_lifetime
        current_radius = int(self.max_radius * progress)
        alpha = int(200 * progress)
        px, py = env._grid_to_pixel(self.pos)
        
        if alpha > 0 and current_radius > 0:
            pygame.gfxdraw.aacircle(surface, px, py, current_radius, (*self.color, alpha))
            pygame.gfxdraw.aacircle(surface, px, py, current_radius-1, (*self.color, alpha))

class PulseEffect(ExplosionEffect):
    def draw(self, surface, env):
        progress = 1.0 - (self.lifetime / self.max_lifetime)
        current_radius = int(self.max_radius * math.sin(progress * math.pi))
        alpha = int(150 * math.sin(progress * math.pi))
        px, py = env._grid_to_pixel(self.pos)
        
        if alpha > 0 and current_radius > 0:
            pygame.gfxdraw.filled_circle(surface, px, py, current_radius, (*self.color, alpha))

class TextPopupEffect(Effect):
    def __init__(self, pos, text, color, lifetime=45, size='normal'):
        super().__init__(pos, lifetime)
        self.text = text
        self.color = color
        self.size = size

    def draw(self, surface, env):
        font = env.font_large if self.size == 'large' else env.font_medium
        text_surf = font.render(self.text, True, self.color)
        
        progress = self.lifetime / self.max_lifetime
        alpha = int(255 * progress)
        text_surf.set_alpha(alpha)
        
        px, py = env._grid_to_pixel(self.pos)
        offset_y = (1.0 - progress) * 30
        
        surface.blit(text_surf, (px - text_surf.get_width()/2, py - text_surf.get_height()/2 - offset_y))

class ChainLinkEffect(Effect):
    def __init__(self, points, color):
        super().__init__((0,0), 30)
        self.points = points
        self.color = color
    
    def draw(self, surface, env):
        progress = self.lifetime / self.max_lifetime
        alpha = int(255 * progress)
        
        pixel_points = [env._grid_to_pixel(p) for p in self.points]
        if len(pixel_points) > 1:
            pygame.draw.aalines(surface, (*self.color, alpha), False, pixel_points, 3)

class WinLossEffect(Effect):
    def __init__(self, text, color):
        super().__init__((0,0), 120)
        self.text = text
        self.color = color

    def draw(self, surface, env):
        progress = 1.0 - (self.lifetime / self.max_lifetime)
        alpha = int(255 * min(1.0, progress * 4))
        
        text_surf = env.font_large.render(self.text, True, self.color)
        text_surf.set_alpha(alpha)
        
        px = env.SCREEN_WIDTH / 2
        py = env.SCREEN_HEIGHT / 3
        surface.blit(text_surf, (px - text_surf.get_width()/2, py - text_surf.get_height()/2))

class UnlockEffect(WinLossEffect):
    def __init__(self, text):
        super().__init__(text, GameEnv.COLOR_CURSOR)
    
    def draw(self, surface, env):
        progress = self.lifetime / self.max_lifetime
        alpha = int(255 * math.sin(progress * math.pi))
        
        text_surf = env.font_medium.render(self.text, True, self.color)
        text_surf.set_alpha(alpha)
        
        px = env.SCREEN_WIDTH / 2
        py = env.SCREEN_HEIGHT / 2
        surface.blit(text_surf, (px - text_surf.get_width()/2, py - text_surf.get_height()/2))

if __name__ == '__main__':
    # Example of how to use the environment
    # This block is for manual play and will not be part of the final environment module
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use a visible driver for manual play
    env = GameEnv(render_mode="rgb_array")
    
    obs, info = env.reset()
    
    pygame.display.set_caption("Quantum Chain Reaction")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    while running:
        movement = 0 # none
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Reward: {reward:.2f}, Score: {info['score']:.0f}")

        if terminated or truncated:
            print("Episode finished!")
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Limit to 30 FPS

    pygame.quit()