import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import random
from collections import namedtuple
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# --- Data Structures for Game Entities ---

Card = namedtuple("Card", ["id", "name", "type", "attack", "defense", "predator_to", "color"])
Zone = namedtuple("Zone", ["id", "rect", "center"])

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A tactical card game of deep-sea conquest. Deploy creatures to control zones "
        "and outmaneuver your opponent to dominate the abyss."
    )
    user_guide = (
        "Controls: Use ←→ to select a zone and ↑↓ to select a card from your hand. "
        "Press space to deploy the card or shift to clone it."
    )
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.MAX_TURNS = 20
        self.MAX_STEPS = 1000 # Safety limit
        self.PLAYER_ID = "player"
        self.AI_ID = "ai"
        self.NEUTRAL_ID = "neutral"
        self.HAND_SIZE = 4
        self.MAX_CREATURES_PER_ZONE = 2
        
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
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_card = pygame.font.SysFont("Consolas", 12)
        self.font_small = pygame.font.SysFont("Consolas", 10)
        self.font_message = pygame.font.SysFont("Consolas", 16)

        # --- Colors ---
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_AI = (255, 50, 50)
        self.COLOR_NEUTRAL = (60, 80, 100)
        self.COLOR_CONTESTED = (255, 200, 0)
        self.COLOR_WHITE = (240, 240, 240)
        self.COLOR_BLACK = (10, 10, 10)
        self.COLOR_SELECTION = (255, 255, 0)
        
        # --- Game Data Definitions ---
        self._define_creatures()
        self._define_zones()
        
        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.turn = 0
        self.score = 0
        self.game_over = False
        self.player_deck = []
        self.player_hand = []
        self.ai_deck = []
        self.ai_hand = []
        self.zone_states = {}
        self.selected_card_idx = 0
        self.selected_zone_idx = 0
        self.particles = []
        self.game_message = ""
        self.last_action_time = 0

        # --- Final setup ---
        if render_mode == "human":
            pygame.display.set_caption("Abyssal Dominion")
            self.human_screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

        self.render_mode = render_mode

    def _define_creatures(self):
        self.CREATURES = {
            0: Card(0, "Goliath Grouper", "Fish", 6, 5, "Crustacean", (200, 100, 50)),
            1: Card(1, "Mantis Shrimp", "Crustacean", 8, 3, "Cephalopod", (50, 200, 100)),
            2: Card(2, "Giant Squid", "Cephalopod", 7, 4, "Fish", (180, 50, 200)),
            3: Card(3, "Barrier Reef", "Structure", 0, 9, None, (150, 150, 150)),
            4: Card(4, "Viperfish", "Fish", 9, 2, "Crustacean", (220, 120, 70)),
            5: Card(5, "Spider Crab", "Crustacean", 4, 7, "Cephalopod", (70, 220, 120)),
        }
        self.INITIAL_DECK_COMPOSITION = [0, 0, 1, 1, 2, 2, 3, 4, 5]

    def _define_zones(self):
        self.ZONES = []
        zone_w, zone_h = 120, 100
        gap_x, gap_y = 20, 30
        start_x = (self.SCREEN_WIDTH - 3 * zone_w - 2 * gap_x) // 2
        start_y = 80
        for i in range(6):
            row, col = divmod(i, 3)
            x = start_x + col * (zone_w + gap_x)
            y = start_y + row * (zone_h + gap_y)
            rect = pygame.Rect(x, y, zone_w, zone_h)
            self.ZONES.append(Zone(id=i, rect=rect, center=rect.center))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.turn = 1
        self.score = 0
        self.game_over = False

        # Create and shuffle decks
        self.player_deck = [self.CREATURES[cid] for cid in self.INITIAL_DECK_COMPOSITION]
        self.ai_deck = [self.CREATURES[cid] for cid in self.INITIAL_DECK_COMPOSITION]
        self.np_random.shuffle(self.player_deck)
        self.np_random.shuffle(self.ai_deck)

        # Draw initial hands
        self.player_hand = [self.player_deck.pop() for _ in range(self.HAND_SIZE) if self.player_deck]
        self.ai_hand = [self.ai_deck.pop() for _ in range(self.HAND_SIZE) if self.ai_deck]

        # Initialize zones
        self.zone_states = {}
        for z in self.ZONES:
            self.zone_states[z.id] = {"controller": self.NEUTRAL_ID, "creatures": []}
        
        self.zone_states[0]["controller"] = self.PLAYER_ID
        self.zone_states[5]["controller"] = self.AI_ID
        
        self.selected_card_idx = 0
        self.selected_zone_idx = 0
        self.particles = []
        self.game_message = "Your Turn"

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        self.game_message = ""

        # --- 1. Player Action Phase ---
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1

        # Handle selections (these happen instantly)
        current_time = pygame.time.get_ticks()
        if current_time - self.last_action_time > 200: # Debounce input
            if movement == 1: self.selected_card_idx = (self.selected_card_idx - 1) % len(self.player_hand) if self.player_hand else 0
            if movement == 2: self.selected_card_idx = (self.selected_card_idx + 1) % len(self.player_hand) if self.player_hand else 0
            if movement == 3: self.selected_zone_idx = (self.selected_zone_idx - 1) % len(self.ZONES)
            if movement == 4: self.selected_zone_idx = (self.selected_zone_idx + 1) % len(self.ZONES)
            if movement != 0:
                self.last_action_time = current_time

        player_acted = False
        # Handle Deploy or Clone
        if space_press or shift_press:
            if self.player_hand:
                selected_card = self.player_hand[self.selected_card_idx]
                
                if space_press: # DEPLOY
                    zone_id = self.selected_zone_idx
                    zone_creatures = self.zone_states[zone_id]["creatures"]
                    if len(zone_creatures) < self.MAX_CREATURES_PER_ZONE:
                        if self.zone_states[zone_id]["controller"] == self.AI_ID:
                            reward += 5.0
                            self.game_message = f"Ambush in Zone {zone_id+1}!"
                        else:
                            self.game_message = f"Deployed {selected_card.name}."
                        
                        self.player_hand.pop(self.selected_card_idx)
                        zone_creatures.append({"owner": self.PLAYER_ID, "card": selected_card})
                        player_acted = True
                    else:
                        self.game_message = "Zone is full!"
                
                elif shift_press: # CLONE
                    self.player_deck.append(selected_card)
                    self.np_random.shuffle(self.player_deck)
                    self.player_hand.pop(self.selected_card_idx)
                    self.game_message = f"Cloned {selected_card.name}."
                    player_acted = True

                if self.player_hand and self.selected_card_idx >= len(self.player_hand):
                    self.selected_card_idx = len(self.player_hand) - 1
            else:
                self.game_message = "No cards to play!"
                player_acted = True # Pass turn if no cards

        # A "pass" action (movement == 0 with no other key) also consumes the turn
        if movement == 0 and not (space_press or shift_press):
            player_acted = True
            self.game_message = "Turn Passed."

        # --- 2. If Player Acted, proceed with game turn ---
        if player_acted:
            self._ai_turn()
            self._resolve_combat()
            reward += self._update_zone_control()
            self._draw_cards()
            self.turn += 1

        # --- 3. Termination Check ---
        terminated = False
        player_zones = sum(1 for z in self.zone_states.values() if z['controller'] == self.PLAYER_ID)
        ai_zones = sum(1 for z in self.zone_states.values() if z['controller'] == self.AI_ID)

        if self.turn > self.MAX_TURNS:
            terminated = True
            if player_zones > ai_zones:
                reward += 50.0
                self.game_message = "Victory! The depths are yours."
            elif ai_zones > player_zones:
                reward -= 50.0
                self.game_message = "Defeat! The ocean rejects you."
            else:
                self.game_message = "Stalemate. The tides are undecided."
        
        if player_zones == 0 and self.turn > 1:
            terminated = True
            reward -= 50.0
            self.game_message = "Defeat! You lost all territory."

        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _ai_turn(self):
        if not self.ai_hand:
            return

        card_to_play = max(self.ai_hand, key=lambda c: c.attack)
        
        target_zones = [zid for zid, z in self.zone_states.items() if z['controller'] == self.NEUTRAL_ID and not z['creatures']]
        if not target_zones:
            target_zones = [zid for zid, z in self.zone_states.items() if z['controller'] == self.AI_ID and len(z['creatures']) < self.MAX_CREATURES_PER_ZONE]
        if not target_zones:
            target_zones = [zid for zid, z in self.zone_states.items() if len(z['creatures']) < self.MAX_CREATURES_PER_ZONE]

        if target_zones:
            zone_id = self.np_random.choice(target_zones)
            self.ai_hand.remove(card_to_play)
            self.zone_states[zone_id]["creatures"].append({"owner": self.AI_ID, "card": card_to_play})

    def _resolve_combat(self):
        for zone_id, state in self.zone_states.items():
            player_creatures = [c for c in state["creatures"] if c["owner"] == self.PLAYER_ID]
            ai_creatures = [c for c in state["creatures"] if c["owner"] == self.AI_ID]

            if player_creatures and ai_creatures:
                p_creature = player_creatures[0]
                ai_creature = ai_creatures[0]
                p_card = p_creature["card"]
                ai_card = ai_creature["card"]

                winner = None
                if p_card.predator_to == ai_card.type:
                    winner = self.PLAYER_ID
                elif ai_card.predator_to == p_card.type:
                    winner = self.AI_ID
                elif p_card.attack > ai_card.defense:
                    winner = self.PLAYER_ID
                elif ai_card.attack > p_card.defense:
                    winner = self.AI_ID
                
                if winner == self.PLAYER_ID:
                    state["creatures"].remove(ai_creature)
                    self._create_particles(self.ZONES[zone_id].center, self.COLOR_PLAYER)
                elif winner == self.AI_ID:
                    state["creatures"].remove(p_creature)
                    self._create_particles(self.ZONES[zone_id].center, self.COLOR_AI)
                else: 
                    state["creatures"].remove(p_creature)
                    state["creatures"].remove(ai_creature)
                    self._create_particles(self.ZONES[zone_id].center, self.COLOR_CONTESTED)

    def _update_zone_control(self):
        reward = 0
        for zone_id, state in self.zone_states.items():
            old_controller = state["controller"]
            
            player_power = sum(1 for c in state["creatures"] if c["owner"] == self.PLAYER_ID)
            ai_power = sum(1 for c in state["creatures"] if c["owner"] == self.AI_ID)

            new_controller = old_controller
            if player_power > ai_power:
                new_controller = self.PLAYER_ID
            elif ai_power > player_power:
                new_controller = self.AI_ID
            elif player_power == 0 and ai_power == 0:
                new_controller = self.NEUTRAL_ID

            if new_controller != old_controller:
                if new_controller == self.PLAYER_ID:
                    reward += 1.0
                elif old_controller == self.PLAYER_ID:
                    reward -= 1.0
                state["controller"] = new_controller
        return reward
    
    def _draw_cards(self):
        while len(self.player_hand) < self.HAND_SIZE and self.player_deck:
            self.player_hand.append(self.player_deck.pop())
        while len(self.ai_hand) < self.HAND_SIZE and self.ai_deck:
            self.ai_hand.append(self.ai_deck.pop())

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "turn": self.turn,
            "player_zones": sum(1 for z in self.zone_states.values() if z['controller'] == self.PLAYER_ID),
            "ai_zones": sum(1 for z in self.zone_states.values() if z['controller'] == self.AI_ID),
        }

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            self.particles.append([list(pos), [vx, vy], random.randint(15, 30), color])

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p[0][0] += p[1][0]
            p[0][1] += p[1][1]
            p[2] -= 1
            if p[2] <= 0:
                self.particles.remove(p)
            else:
                radius = max(0, p[2] // 4)
                pygame.draw.circle(self.screen, p[3], (int(p[0][0]), int(p[0][1])), radius)

    def _render_all(self):
        self.screen.fill(self.COLOR_BG)
        for i, zone in enumerate(self.ZONES):
            self._render_zone(zone, i == self.selected_zone_idx)
        self._update_and_draw_particles()
        self._render_hand()
        self._render_ui()

    def _render_zone(self, zone, is_selected):
        state = self.zone_states[zone.id]
        controller = state["controller"]
        
        player_present = any(c['owner'] == self.PLAYER_ID for c in state['creatures'])
        ai_present = any(c['owner'] == self.AI_ID for c in state['creatures'])

        if player_present and ai_present:
            color = self.COLOR_CONTESTED
        elif controller == self.PLAYER_ID:
            color = self.COLOR_PLAYER
        elif controller == self.AI_ID:
            color = self.COLOR_AI
        else:
            color = self.COLOR_NEUTRAL

        pygame.draw.rect(self.screen, color, zone.rect, border_radius=10)
        pygame.draw.rect(self.screen, tuple(min(255, c+40) for c in color), zone.rect, width=2, border_radius=10)

        if is_selected:
            pygame.draw.rect(self.screen, self.COLOR_SELECTION, zone.rect, width=4, border_radius=12)
        
        creature_rects = [
            pygame.Rect(zone.rect.left + 5, zone.rect.top + 5, 50, 90),
            pygame.Rect(zone.rect.right - 55, zone.rect.top + 5, 50, 90)
        ]
        for creature in state["creatures"]:
            if creature_rects:
                rect = creature_rects.pop(0)
                self._render_card_in_zone(creature["card"], creature["owner"], rect)

    def _render_card_in_zone(self, card, owner, rect):
        color = self.COLOR_PLAYER if owner == self.PLAYER_ID else self.COLOR_AI
        pygame.draw.rect(self.screen, self.COLOR_BLACK, rect, border_radius=5)
        pygame.draw.rect(self.screen, color, rect, width=2, border_radius=5)
        
        name_surf = self.font_small.render(card.name, True, self.COLOR_WHITE)
        self.screen.blit(name_surf, (rect.centerx - name_surf.get_width()//2, rect.top + 5))

        pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, 15, card.color)
        pygame.gfxdraw.aacircle(self.screen, rect.centerx, rect.centery, 15, self.COLOR_WHITE)

        stats_surf = self.font_card.render(f"{card.attack} / {card.defense}", True, self.COLOR_WHITE)
        self.screen.blit(stats_surf, (rect.centerx - stats_surf.get_width()//2, rect.bottom - 20))

    def _render_hand(self):
        if not self.player_hand:
            return
            
        card_w, card_h = 80, 110
        total_width = len(self.player_hand) * card_w + (len(self.player_hand) - 1) * 10
        start_x = (self.SCREEN_WIDTH - total_width) // 2

        for i, card in enumerate(self.player_hand):
            rect = pygame.Rect(start_x + i * (card_w + 10), self.SCREEN_HEIGHT - card_h - 10, card_w, card_h)
            self._render_card_in_hand(card, rect, i == self.selected_card_idx)

    def _render_card_in_hand(self, card, rect, is_selected):
        base_color = self.COLOR_SELECTION if is_selected else self.COLOR_PLAYER
        pygame.draw.rect(self.screen, self.COLOR_BLACK, rect, border_radius=8)
        pygame.draw.rect(self.screen, base_color, rect, width=3 if is_selected else 2, border_radius=8)
        
        name_surf = self.font_card.render(card.name, True, self.COLOR_WHITE)
        self.screen.blit(name_surf, (rect.centerx - name_surf.get_width()//2, rect.top + 8))

        pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery - 5, 20, card.color)
        pygame.gfxdraw.aacircle(self.screen, rect.centerx, rect.centery - 5, 20, self.COLOR_WHITE)

        stats_surf = self.font_card.render(f"ATK: {card.attack}", True, self.COLOR_WHITE)
        self.screen.blit(stats_surf, (rect.centerx - stats_surf.get_width()//2, rect.bottom - 35))
        stats_surf = self.font_card.render(f"DEF: {card.defense}", True, self.COLOR_WHITE)
        self.screen.blit(stats_surf, (rect.centerx - stats_surf.get_width()//2, rect.bottom - 20))
        
        if card.predator_to:
            pred_surf = self.font_small.render(f"vs {card.predator_to}", True, self.COLOR_CONTESTED)
            self.screen.blit(pred_surf, (rect.centerx - pred_surf.get_width()//2, rect.top + 25))

    def _render_ui(self):
        score_text = self.font_main.render(f"Score: {int(self.score)}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (10, 10))
        
        turn_text = self.font_main.render(f"Turn: {self.turn}/{self.MAX_TURNS}", True, self.COLOR_WHITE)
        self.screen.blit(turn_text, (self.SCREEN_WIDTH - turn_text.get_width() - 10, 10))

        player_deck_text = self.font_card.render(f"Deck: {len(self.player_deck)}", True, self.COLOR_PLAYER)
        self.screen.blit(player_deck_text, (10, self.SCREEN_HEIGHT - 25))
        ai_deck_text = self.font_card.render(f"AI Hand: {len(self.ai_hand)}", True, self.COLOR_AI)
        self.screen.blit(ai_deck_text, (self.SCREEN_WIDTH - ai_deck_text.get_width() - 10, self.SCREEN_HEIGHT - 25))

        if self.game_message:
            msg_surf = self.font_message.render(self.game_message, True, self.COLOR_SELECTION)
            self.screen.blit(msg_surf, (self.SCREEN_WIDTH//2 - msg_surf.get_width()//2, 45))

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_observation()
        elif self.render_mode == "human":
            if not hasattr(self, 'human_screen'):
                 self.human_screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            self.human_screen.blit(self.screen, (0, 0))
            pygame.display.flip()

    def close(self):
        pygame.quit()

# --- Example Usage ---
if __name__ == "__main__":
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    
    print("--- Abyssal Dominion ---")
    print(GameEnv.game_description)
    print(GameEnv.user_guide)
    print("Goal: Control more zones than the AI after 20 turns.")

    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op / pass turn
        
        # In a real scenario, you'd get actions from a policy.
        # For manual play, we check for key presses.
        # A single key press should correspond to one action, then a step.
        
        event_happened = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                event_happened = True
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1
                elif event.key == pygame.K_p: # Explicit pass
                    action = [0, 0, 0]
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    event_happened = False # Don't step after reset
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        # Only step if an action was taken
        if event_happened:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Turn: {info['turn']}, Score: {info['score']:.1f}, Reward: {reward:.1f}, Terminated: {terminated}")
            
            if terminated or truncated:
                print("Game Over!")
                # Optional: wait for a key press to reset
                # input("Press Enter to play again...")
                obs, info = env.reset()

        env.render()
        env.clock.tick(30) # Limit FPS

    env.close()