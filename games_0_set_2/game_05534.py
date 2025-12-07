import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import namedtuple
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# --- Data Structures ---
Card = namedtuple("Card", ["id", "name", "attack", "defense", "effect"])
Particle = namedtuple("Particle", ["x", "y", "vx", "vy", "life", "color", "size"])

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to select a card. Space to play the selected card. "
        "The game is turn-based."
    )

    game_description = (
        "A strategic 1v1 card duel. Play cards into lanes to attack your "
        "opponent. Reduce their health to zero to win."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_m = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 32, bold=True)

        # --- Game Constants ---
        self.MAX_HP = 20
        self.HAND_SIZE = 5
        self.NUM_LANES = 5
        self.MAX_STEPS = 1000

        # --- Visuals & Colors ---
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_OPPONENT = (255, 80, 80)
        self.COLOR_HEALTH = (100, 220, 100)
        self.COLOR_DAMAGE = (255, 200, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (10, 10, 10)
        self.COLOR_CARD_BG = (60, 70, 80)
        self.COLOR_CARD_BORDER = (90, 100, 110)
        self.COLOR_SELECT_GLOW = (255, 255, 100)

        # --- State Initialization ---
        self._create_master_deck()
        self.particles = []
        self.animating_cards = []
        # The reset method is called last to ensure all attributes are initialized
        # self.reset() is not called here to allow for seeding in reset()
        
    def _create_master_deck(self):
        """Creates the set of 20 unique cards for the game."""
        self.master_deck = [
            Card(0, "Rookie", 2, 1, None),
            Card(1, "Guard", 1, 3, "heal_1"),
            Card(2, "Brawler", 3, 1, None),
            Card(3, "Knight", 3, 3, None),
            Card(4, "Giant", 2, 5, None),
            Card(5, "Berserker", 5, 1, "self_dmg_1"),
            Card(6, "Scout", 2, 2, None),
            Card(7, "Archer", 4, 1, None),
            Card(8, "Swordsman", 4, 2, None),
            Card(9, "Pikeman", 2, 4, None),
            Card(10, "Assassin", 5, 2, None),
            Card(11, "Mage", 4, 3, "direct_dmg_1"),
            Card(12, "Healer", 1, 2, "heal_2"),
            Card(13, "Vanguard", 3, 4, None),
            Card(14, "Champion", 5, 4, None),
            Card(15, "Paladin", 4, 5, "heal_1"),
            Card(16, "Warlock", 6, 2, "self_dmg_1"),
            Card(17, "Golem", 3, 6, None),
            Card(18, "Dragon", 7, 5, None),
            Card(19, "King", 6, 6, None),
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.deck = self.master_deck.copy()
        self.np_random.shuffle(self.deck)

        self.player_hp = self.MAX_HP
        self.opponent_hp = self.MAX_HP
        self.player_display_hp = self.MAX_HP
        self.opponent_display_hp = self.MAX_HP

        self.player_hand = [self._draw_card() for _ in range(self.HAND_SIZE)]
        self.opponent_hand = [self._draw_card() for _ in range(self.HAND_SIZE)]

        self.battlefield = [[None, None] for _ in range(self.NUM_LANES)]
        self.selected_card_index = 0
        self.turn_phase = "PLAYER_TURN"
        self.turn_message = "Your Turn"

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles.clear()
        self.animating_cards.clear()

        return self._get_observation(), self._get_info()

    def _draw_card(self):
        if not self.deck:
            return None # No cards left to draw
        return self.deck.pop(0)

    def step(self, action):
        if self.game_over:
            # Re-render to show final state if needed, then return
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        move_action, space_action, _ = action
        space_pressed = space_action == 1

        if self.turn_phase == "PLAYER_TURN":
            if not self.player_hand: # Player has no cards, must pass
                space_pressed = True

            # --- Handle Player Input ---
            if move_action == 3:  # Left
                if self.player_hand:
                    self.selected_card_index = (self.selected_card_index - 1) % len(self.player_hand)
            elif move_action == 4:  # Right
                if self.player_hand:
                    self.selected_card_index = (self.selected_card_index + 1) % len(self.player_hand)
            
            if space_pressed:
                # --- Player Action Phase ---
                played_card = None
                lane_idx = None  # Initialize lane_idx to handle cases where no card is played
                if self.player_hand:
                    lane_idx = self._find_next_open_lane()
                    if lane_idx is not None:
                        played_card = self.player_hand.pop(self.selected_card_index)
                        self.battlefield[lane_idx][0] = played_card
                        self.selected_card_index = max(0, min(self.selected_card_index, len(self.player_hand) - 1))
                        self._add_card_animation(played_card, 'player', lane_idx)
                        
                        # Reward for card choice
                        reward += self._calculate_play_reward(played_card, self.player_hand)

                # --- Opponent Action Phase ---
                opponent_card = self._opponent_ai_play(lane_idx)
                if opponent_card and lane_idx is not None:
                    self.battlefield[lane_idx][1] = opponent_card
                    self._add_card_animation(opponent_card, 'opponent', lane_idx)

                # --- Combat Resolution Phase ---
                if lane_idx is not None:
                    combat_reward = self._resolve_combat(lane_idx)
                    reward += combat_reward
                
                # --- Draw Phase ---
                if len(self.player_hand) < self.HAND_SIZE:
                    new_card = self._draw_card()
                    if new_card: self.player_hand.append(new_card)
                
                if len(self.opponent_hand) < self.HAND_SIZE:
                    new_card = self._draw_card()
                    if new_card: self.opponent_hand.append(new_card)
        
        self._update_animations()
        self.steps += 1
        self.score += reward
        terminated = self._check_termination()
        
        if terminated:
            terminal_reward = 0
            if self.player_hp <= 0 and self.opponent_hp > 0:
                terminal_reward = -100
                self.turn_message = "You Lose!"
            elif self.opponent_hp <= 0 and self.player_hp > 0:
                terminal_reward = 100
                self.turn_message = "You Win!"
            else: # Draw or max steps
                self.turn_message = "Game Over"
            reward += terminal_reward
            self.score += terminal_reward
            self.game_over = True

        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True # Per Gymnasium API, if truncated, terminated must be True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _find_next_open_lane(self):
        for i in range(self.NUM_LANES):
            if self.battlefield[i][0] is None and self.battlefield[i][1] is None:
                return i
        return None

    def _calculate_play_reward(self, played_card, remaining_hand):
        if not remaining_hand:
            return 0
        avg_power = sum(c.attack + c.defense for c in remaining_hand) / len(remaining_hand)
        played_power = played_card.attack + played_card.defense
        if played_power < avg_power * 0.8:
            return -0.2 # Penalty for playing a weak card
        return 0

    def _opponent_ai_play(self, lane_idx):
        if not self.opponent_hand or lane_idx is None:
            return None

        # Simple AI: play offensively if winning, defensively if losing
        if self.opponent_hp > self.player_hp:
            # Play highest attack card
            card_to_play = max(self.opponent_hand, key=lambda c: c.attack)
        else:
            # Play highest defense card
            card_to_play = max(self.opponent_hand, key=lambda c: c.defense)
        
        self.opponent_hand.remove(card_to_play)
        return card_to_play

    def _resolve_combat(self, lane_idx):
        player_card, opponent_card = self.battlefield[lane_idx]
        reward = 0

        if player_card is None and opponent_card is None:
            return 0

        # --- Handle Special Effects ---
        if player_card and player_card.effect:
            reward += self._apply_effect(player_card.effect, 'player')
        if opponent_card and opponent_card.effect:
            reward -= self._apply_effect(opponent_card.effect, 'opponent')

        # --- Handle Combat Damage ---
        player_atk = player_card.attack if player_card else 0
        opponent_atk = opponent_card.attack if opponent_card else 0
        player_def = player_card.defense if player_card else 0
        opponent_def = opponent_card.defense if opponent_card else 0

        px, py = self._get_lane_pos(lane_idx, 'player')
        ox, oy = self._get_lane_pos(lane_idx, 'opponent')
        self._create_particles((px + ox) // 2, (py + oy) // 2, 20, self.COLOR_DAMAGE)

        if player_atk > opponent_def:
            damage = player_atk - opponent_def
            self.opponent_hp -= damage
            reward += damage * 0.1 # Reward for dealing damage
        if opponent_atk > player_def:
            damage = opponent_atk - player_def
            self.player_hp -= damage
            reward -= damage * 0.1 # Penalty for taking damage
        
        self.battlefield[lane_idx] = [None, None] # Clear lane after combat
        return reward

    def _apply_effect(self, effect, caster):
        reward = 0
        if effect == "heal_1":
            if caster == 'player': self.player_hp = min(self.MAX_HP, self.player_hp + 1); reward = 0.5
            else: self.opponent_hp = min(self.MAX_HP, self.opponent_hp + 1)
        elif effect == "heal_2":
            if caster == 'player': self.player_hp = min(self.MAX_HP, self.player_hp + 2); reward = 1.0
            else: self.opponent_hp = min(self.MAX_HP, self.opponent_hp + 2)
        elif effect == "direct_dmg_1":
            if caster == 'player': self.opponent_hp -= 1; reward = 1.0
            else: self.player_hp -= 1
        elif effect == "self_dmg_1":
            if caster == 'player': self.player_hp -= 1; reward = -1.0
            else: self.opponent_hp -= 1
        return reward

    def _check_termination(self):
        return self.player_hp <= 0 or self.opponent_hp <= 0

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "player_hp": self.player_hp, "opponent_hp": self.opponent_hp}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    # --- Update and Animation Logic ---

    def _update_animations(self):
        # Smooth HP bars
        self.player_display_hp += (self.player_hp - self.player_display_hp) * 0.1
        self.opponent_display_hp += (self.opponent_hp - self.opponent_display_hp) * 0.1

        # Update particles
        new_particles = []
        for p in self.particles:
            if p.life > 0:
                new_particles.append(Particle(p.x + p.vx, p.y + p.vy, p.vx, p.vy, p.life - 1, p.color, p.size))
        self.particles = new_particles

        # Update card animations
        new_animating_cards = []
        for card, owner, lane_idx, timer, start_pos, end_pos in self.animating_cards:
            if timer > 0:
                new_animating_cards.append((card, owner, lane_idx, timer - 1, start_pos, end_pos))
        self.animating_cards = new_animating_cards

    def _add_card_animation(self, card, owner, lane_idx):
        if owner == 'player':
            start_pos = (self.WIDTH // 2, self.HEIGHT - 50)
        else: # opponent
            start_pos = (self.WIDTH // 2, 50)
        end_pos = self._get_lane_pos(lane_idx, owner)
        self.animating_cards.append((card, owner, lane_idx, 15, start_pos, end_pos)) # 15 frames animation

    def _create_particles(self, x, y, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            life = self.np_random.integers(10, 20)
            size = self.np_random.uniform(1, 4)
            self.particles.append(Particle(x, y, vx, vy, life, color, size))

    # --- Rendering Logic ---

    def _render_game(self):
        # Draw battlefield lanes
        for i in range(self.NUM_LANES):
            x, _ = self._get_lane_pos(i, 'player')
            pygame.draw.rect(self.screen, self.COLOR_GRID, (x - 45, 100, 90, self.HEIGHT - 200), border_radius=5)

        # Draw cards on battlefield (non-animating)
        for i in range(self.NUM_LANES):
            is_animating_p = any(ac[1] == 'player' and ac[2] == i for ac in self.animating_cards)
            is_animating_o = any(ac[1] == 'opponent' and ac[2] == i for ac in self.animating_cards)
            
            if self.battlefield[i][0] and not is_animating_p:
                x, y = self._get_lane_pos(i, 'player')
                self._render_card(self.battlefield[i][0], x, y)
            if self.battlefield[i][1] and not is_animating_o:
                x, y = self._get_lane_pos(i, 'opponent')
                self._render_card(self.battlefield[i][1], x, y)
        
        # Draw animating cards
        for card, owner, lane_idx, timer, start_pos, end_pos in self.animating_cards:
            progress = 1.0 - (timer / 15.0)
            curr_x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
            curr_y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress
            self._render_card(card, curr_x, curr_y)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p.life / 20))))
            try:
                # Use a surface for alpha blending
                s = pygame.Surface((p.size*2, p.size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*p.color, alpha), (p.size, p.size), p.size)
                self.screen.blit(s, (p.x - p.size, p.y - p.size))
            except (ValueError, TypeError):
                 # Fallback for invalid alpha or color
                 pygame.draw.circle(self.screen, p.color, (int(p.x), int(p.y)), int(p.size))


    def _render_ui(self):
        # --- Player and Opponent Info ---
        self._render_text("Player", self.font_m, 60, self.HEIGHT - 80)
        self._render_hp_bar(self.player_display_hp, self.MAX_HP, 60, self.HEIGHT - 60, self.COLOR_PLAYER)
        
        self._render_text("Opponent", self.font_m, 60, 40)
        self._render_hp_bar(self.opponent_display_hp, self.MAX_HP, 60, 60, self.COLOR_OPPONENT)
        
        # --- Turn Indicator ---
        if not self.game_over:
            self._render_text(self.turn_message, self.font_m, self.WIDTH // 2, self.HEIGHT // 2)
        else:
            self._render_text(self.turn_message, self.font_l, self.WIDTH // 2, self.HEIGHT // 2, self.COLOR_SELECT_GLOW)


        # --- Player Hand ---
        if self.player_hand:
            num_cards = len(self.player_hand)
            card_spacing = 90
            start_x = self.WIDTH // 2 - (num_cards - 1) * card_spacing // 2
            for i, card in enumerate(self.player_hand):
                x = start_x + i * card_spacing
                y = self.HEIGHT - 50
                is_selected = (i == self.selected_card_index) and (self.turn_phase == "PLAYER_TURN")
                self._render_card(card, x, y, is_selected)
        else:
            self._render_text("No cards in hand", self.font_m, self.WIDTH // 2, self.HEIGHT - 50)
            
        # --- Score and Steps ---
        self._render_text(f"Score: {self.score:.1f}", self.font_s, self.WIDTH - 70, 20)
        self._render_text(f"Step: {self.steps}/{self.MAX_STEPS}", self.font_s, self.WIDTH - 70, 40)

    def _render_card(self, card, x, y, selected=False):
        card_w, card_h = 80, 50
        rect = pygame.Rect(x - card_w // 2, y - card_h // 2, card_w, card_h)
        
        pygame.draw.rect(self.screen, self.COLOR_CARD_BG, rect, border_radius=5)
        border_color = self.COLOR_SELECT_GLOW if selected else self.COLOR_CARD_BORDER
        border_width = 3 if selected else 2
        pygame.draw.rect(self.screen, border_color, rect, border_width, border_radius=5)
        
        if selected: # Glow effect
            glow_rect = rect.inflate(6, 6)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, (*self.COLOR_SELECT_GLOW, 50), s.get_rect(), 4, border_radius=8)
            self.screen.blit(s, glow_rect.topleft)

        # Attack and Defense values
        self._render_text(f"{card.attack}", self.font_m, x - 25, y, self.COLOR_OPPONENT)
        self._render_text(f"{card.defense}", self.font_m, x + 25, y, self.COLOR_PLAYER)
        
        # Effect indicator
        if card.effect:
            pygame.draw.circle(self.screen, self.COLOR_SELECT_GLOW, (int(x), int(y - 15)), 3)


    def _render_hp_bar(self, current_hp, max_hp, x, y, color):
        bar_w, bar_h = 150, 20
        health_frac = max(0, current_hp / max_hp)
        
        bg_rect = pygame.Rect(x - bar_w // 2, y - bar_h // 2, bar_w, bar_h)
        pygame.draw.rect(self.screen, self.COLOR_GRID, bg_rect, border_radius=5)
        
        fill_rect = pygame.Rect(x - bar_w // 2, y - bar_h // 2, int(bar_w * health_frac), bar_h)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH, fill_rect, border_radius=5)
        
        pygame.draw.rect(self.screen, color, bg_rect, 2, border_radius=5)

        hp_text = f"{max(0, math.ceil(current_hp))}/{max_hp}"
        self._render_text(hp_text, self.font_s, x, y)

    def _render_text(self, text, font, x, y, color=None, shadow_color=None):
        if color is None: color = self.COLOR_TEXT
        if shadow_color is None: shadow_color = self.COLOR_TEXT_SHADOW
        
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect(center=(x, y))
        
        shadow_surf = font.render(text, True, shadow_color)
        shadow_rect = shadow_surf.get_rect(center=(x + 1, y + 1))

        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def _get_lane_pos(self, lane_idx, owner):
        x = self.WIDTH // 2 + (lane_idx - (self.NUM_LANES - 1) / 2) * 100
        y = self.HEIGHT // 2 + (70 if owner == 'player' else -70)
        return int(x), int(y)

# --- Example Usage (for testing) ---
if __name__ == "__main__":
    # The main loop is for human play and visualization, not for training.
    # It sets up a pygame window and captures keyboard input.
    
    # Re-enable video driver for human play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset(seed=42)
    done = False
    
    pygame.display.set_caption("Card Duel")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    action = env.action_space.sample() 
    action.fill(0) 

    while not done:
        # --- Human Input ---
        # This event loop is for manual control. 
        # An agent would not use this loop.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                action.fill(0) # Reset action on new key press
                if event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                
                # --- Environment Step on key press ---
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")


        # --- Render to screen ---
        # The observation is the game's rendered state.
        # We convert it back to a surface to display it.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS for human play
    
    print(f"Game Over. Final Score: {info['score']}, Steps: {info['steps']}")
    pygame.quit()