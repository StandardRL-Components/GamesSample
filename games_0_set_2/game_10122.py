import gymnasium as gym
import os
import pygame
import numpy as np
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:57:23.654090
# Source Brief: brief_00122.md
# Brief Index: 122
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw


class GameEnv(gym.Env):
    """
    Gymnasium environment: Defend your mainframe from cascading viral attacks by
    strategically placing firewall upgrade cards in a physics-based environment.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your mainframe from cascading viral attacks by strategically placing firewall "
        "upgrade cards in a physics-based environment."
    )
    user_guide = (
        "Controls: Use 1-4 or ←↑↓→ to select a slot, Shift to cycle through cards, and Space to place a card."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    WIN_CONDITION_STEPS = 500
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_GRID = (20, 30, 50)
    COLOR_MAINFRAME = (0, 100, 255)
    COLOR_MAINFRAME_GLOW = (0, 50, 150)
    COLOR_VIRUS = (255, 0, 80)
    COLOR_VIRUS_GLOW = (150, 0, 40)
    COLOR_HEALTH_BAR = (0, 255, 120)
    COLOR_HEALTH_BAR_BG = (80, 0, 0)
    COLOR_UI_TEXT = (200, 220, 255)
    COLOR_SLOT = (40, 60, 100)
    COLOR_SLOT_SELECTED = (255, 255, 0)
    
    CARD_COLORS = {
        "REFLECTOR": (0, 150, 255),
        "DAMAGER": (255, 100, 0),
        "SLOWER": (150, 0, 255),
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.mainframe_rect = None
        self.mainframe_health = 0
        self.mainframe_max_health = 0
        self.viruses = []
        self.particles = []
        self.card_slots = []
        self.cards_in_slots = {}
        self.player_hand = []
        self.selected_card_idx = 0
        self.selected_slot_idx = 0
        self.base_virus_speed = 0
        self.virus_spawn_chance = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.last_reward_info = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.mainframe_max_health = 100
        self.mainframe_health = self.mainframe_max_health
        self.mainframe_rect = pygame.Rect(
            self.SCREEN_WIDTH // 2 - 50, self.SCREEN_HEIGHT // 2 - 25, 100, 50
        )
        
        self.viruses = []
        self.particles = []
        
        slot_width, slot_height = 60, 80
        slot_y = self.SCREEN_HEIGHT - slot_height - 10
        total_slot_width = 4 * slot_width + 3 * 20
        start_x = (self.SCREEN_WIDTH - total_slot_width) // 2
        self.card_slots = [
            pygame.Rect(start_x + i * (slot_width + 20), slot_y, slot_width, slot_height)
            for i in range(4)
        ]
        self.cards_in_slots = {}
        
        self.player_hand = self._deal_new_hand(3)
        self.selected_card_idx = 0
        self.selected_slot_idx = 0
        
        self.base_virus_speed = 1.0
        self.virus_spawn_chance = 0.015

        self.prev_space_held = True # Avoid action on first frame
        self.prev_shift_held = True

        self.last_reward_info = {}
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.01  # Survival reward
        self.last_reward_info.clear()

        # 1. Handle Player Actions
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held, shift_held)
        
        # 2. Update Game Logic
        self._update_viruses()
        self._update_particles()
        
        # 3. Collision Detection
        for virus in self.viruses[:]: # Iterate over a copy
            # Virus vs. Mainframe
            if self.mainframe_rect.colliderect(virus.rect):
                damage = 10
                self.mainframe_health -= damage
                self.mainframe_health = max(0, self.mainframe_health)
                reward -= 5.0
                self.last_reward_info['mainframe_hit'] = -5.0
                self._create_particles(virus.pos, self.COLOR_VIRUS, 20)
                # sfx: mainframe_damage.wav
                self.viruses.remove(virus)
                continue

            # Virus vs. Cards
            for i, card_rect in enumerate(self.card_slots):
                if i in self.cards_in_slots and card_rect.colliderect(virus.rect):
                    card = self.cards_in_slots[i]
                    card_destroyed_virus = card.activate(virus, self)
                    if card_destroyed_virus:
                        reward += 1.0
                        self.last_reward_info['virus_destroyed'] = self.last_reward_info.get('virus_destroyed', 0) + 1.0
                        self.score += 100
                    break # Virus interacts with one card per step

        # 4. Spawn New Viruses
        self._spawn_viruses()

        # 5. Update Game State
        self.steps += 1
        self._update_difficulty()
        
        # 6. Check Termination
        terminated = self.mainframe_health <= 0
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True
            if self.mainframe_health > 0 and self.steps >= self.WIN_CONDITION_STEPS:
                reward += 100.0
                self.last_reward_info['win_bonus'] = 100.0
                self.score += 10000

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Movement selects a slot (1-4 maps to index 0-3)
        if 1 <= movement <= 4:
            self.selected_slot_idx = movement - 1

        # Shift cycles through cards in hand (on button press)
        if shift_held and not self.prev_shift_held:
            if self.player_hand:
                self.selected_card_idx = (self.selected_card_idx + 1) % len(self.player_hand)
            # sfx: ui_cycle.wav

        # Space places the selected card (on button press)
        if space_held and not self.prev_space_held:
            if self.selected_slot_idx not in self.cards_in_slots and self.player_hand:
                card_type_to_place = self.player_hand.pop(self.selected_card_idx)
                card_class = globals()[f"{card_type_to_place.capitalize()}Card"]
                self.cards_in_slots[self.selected_slot_idx] = card_class(self.card_slots[self.selected_slot_idx])
                
                # Deal a new card to replace the used one
                self.player_hand.extend(self._deal_new_hand(1))
                self.selected_card_idx = min(self.selected_card_idx, len(self.player_hand) - 1) if self.player_hand else 0
                # sfx: card_place.wav
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_viruses(self):
        for virus in self.viruses:
            virus.update()
            # Bounce off screen edges (except bottom)
            if virus.pos[0] <= virus.radius or virus.pos[0] >= self.SCREEN_WIDTH - virus.radius:
                virus.vel[0] *= -1
            if virus.pos[1] <= virus.radius:
                virus.vel[1] *= -1

    def _update_particles(self):
        for p in self.particles[:]:
            p.update()
            if p.lifespan <= 0:
                self.particles.remove(p)

    def _spawn_viruses(self):
        if self.np_random.random() < self.virus_spawn_chance:
            spawn_edge = self.np_random.integers(0, 3) # 0: top, 1: left, 2: right
            pos = [0, 0]
            if spawn_edge == 0: # Top
                pos = [self.np_random.uniform(20, self.SCREEN_WIDTH - 20), 20]
            elif spawn_edge == 1: # Left
                pos = [20, self.np_random.uniform(20, self.SCREEN_HEIGHT - 120)]
            else: # Right
                pos = [self.SCREEN_WIDTH - 20, self.np_random.uniform(20, self.SCREEN_HEIGHT - 120)]
            
            self.viruses.append(Virus(pos, self.mainframe_rect.center, self.base_virus_speed, self.np_random))

    def _update_difficulty(self):
        # Increase virus speed every 50 steps
        if self.steps > 0 and self.steps % 50 == 0:
            self.base_virus_speed += 0.05
        # Increase spawn frequency every 100 steps
        if self.steps > 0 and self.steps % 100 == 0:
            self.virus_spawn_chance *= 1.01

    def _deal_new_hand(self, num_cards):
        return self.np_random.choice(list(self.CARD_COLORS.keys()), num_cards).tolist()

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "mainframe_health": self.mainframe_health,
            "viruses_on_screen": len(self.viruses),
            "last_reward_breakdown": self.last_reward_info
        }
        
    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 1)
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y), 1)

    def _render_game(self):
        # Render Mainframe
        self._render_glow_rect(self.mainframe_rect, self.COLOR_MAINFRAME, self.COLOR_MAINFRAME_GLOW)
        
        # Render Cards in Slots
        for i, card in self.cards_in_slots.items():
            card.draw(self.screen)

        # Render Viruses
        for virus in self.viruses:
            virus.draw(self.screen, self.COLOR_VIRUS, self.COLOR_VIRUS_GLOW)
            
        # Render Particles
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Render Card Slots
        for i, slot_rect in enumerate(self.card_slots):
            color = self.COLOR_SLOT_SELECTED if i == self.selected_slot_idx else self.COLOR_SLOT
            pygame.draw.rect(self.screen, color, slot_rect, 2, border_radius=5)

        # Render Player Hand
        hand_title = self.font_small.render("CARD HAND", True, self.COLOR_UI_TEXT)
        self.screen.blit(hand_title, (20, self.SCREEN_HEIGHT - 100))
        for i, card_type in enumerate(self.player_hand):
            card_rect = pygame.Rect(20, self.SCREEN_HEIGHT - 80 + i * 25, 100, 20)
            color = self.CARD_COLORS[card_type]
            is_selected = (i == self.selected_card_idx)
            
            if is_selected:
                pygame.draw.rect(self.screen, self.COLOR_SLOT_SELECTED, card_rect.inflate(8, 8), 0, border_radius=5)

            pygame.draw.rect(self.screen, color, card_rect, 0, border_radius=5)
            card_text = self.font_small.render(card_type, True, self.COLOR_BG)
            text_rect = card_text.get_rect(center=card_rect.center)
            self.screen.blit(card_text, text_rect)

        # Render Score and Steps
        steps_text = self.font_large.render(f"STEP: {self.steps}/{self.WIN_CONDITION_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (10, 10))
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)

        # Render Mainframe Health
        health_ratio = self.mainframe_health / self.mainframe_max_health
        health_bar_rect = pygame.Rect(
            self.mainframe_rect.left, self.mainframe_rect.top - 15, self.mainframe_rect.width, 10
        )
        health_fill_rect = pygame.Rect(
            health_bar_rect.left, health_bar_rect.top, health_bar_rect.width * health_ratio, health_bar_rect.height
        )
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, health_bar_rect, 0, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, health_fill_rect, 0, border_radius=3)
        health_text = self.font_small.render(f"{self.mainframe_health}/{self.mainframe_max_health}", True, self.COLOR_UI_TEXT)
        health_text_rect = health_text.get_rect(center=health_bar_rect.center)
        self.screen.blit(health_text, health_text_rect)

    def _render_glow_rect(self, rect, color, glow_color):
        glow_size = 15
        s = pygame.Surface((rect.width + glow_size * 2, rect.height + glow_size * 2), pygame.SRCALPHA)
        for i in range(glow_size, 0, -1):
            alpha = int(100 * (1 - i / glow_size))
            temp_glow_color = glow_color + (alpha,)
            pygame.draw.rect(s, temp_glow_color, s.get_rect().inflate(-i*2, -i*2), border_radius=15)
        self.screen.blit(s, (rect.left - glow_size, rect.top - glow_size))
        pygame.draw.rect(self.screen, color, rect, 0, border_radius=8)
        pygame.draw.rect(self.screen, tuple(min(255, c+50) for c in color), rect, 2, border_radius=8)
        
    def _create_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append(Particle(pos, color, self.np_random))

# --- Helper Classes ---

class Virus:
    def __init__(self, pos, target_pos, speed, np_random):
        self.pos = np.array(pos, dtype=float)
        self.radius = 8
        self.health = 20
        self.max_health = 20
        self.np_random = np_random
        
        direction = np.array(target_pos, dtype=float) - self.pos
        norm = np.linalg.norm(direction)
        if norm > 0:
            self.vel = (direction / norm) * speed
        else:
            self.vel = np.array([0, speed])
        self.rect = pygame.Rect(0, 0, self.radius * 2, self.radius * 2)
        self.rect.center = self.pos

    def update(self):
        self.pos += self.vel
        self.rect.center = (int(self.pos[0]), int(self.pos[1]))

    def draw(self, surface, color, glow_color):
        # Glow effect
        glow_radius = int(self.radius * 1.8)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(s, glow_radius, glow_radius, glow_radius, glow_color + (80,))
        surface.blit(s, (self.rect.centerx - glow_radius, self.rect.centery - glow_radius))

        # Main circle
        pygame.gfxdraw.filled_circle(surface, self.rect.centerx, self.rect.centery, self.radius, color)
        pygame.gfxdraw.aacircle(surface, self.rect.centerx, self.rect.centery, self.radius, color)

class Particle:
    def __init__(self, pos, color, np_random):
        self.pos = np.array(pos, dtype=float)
        angle = np_random.uniform(0, 2 * math.pi)
        speed = np_random.uniform(1, 4)
        self.vel = np.array([math.cos(angle), math.sin(angle)]) * speed
        self.lifespan = np_random.integers(15, 30)
        self.color = color
        self.radius = self.lifespan / 8

    def update(self):
        self.pos += self.vel
        self.vel *= 0.95 # friction
        self.lifespan -= 1
        self.radius = max(0, self.lifespan / 8)

    def draw(self, surface):
        if self.lifespan > 0:
            alpha = int(255 * (self.lifespan / 30))
            color_with_alpha = self.color + (alpha,)
            pygame.draw.circle(surface, color_with_alpha, (int(self.pos[0]), int(self.pos[1])), int(self.radius))

class Card:
    def __init__(self, rect, card_type):
        self.rect = rect
        self.type = card_type
        self.color = GameEnv.CARD_COLORS[card_type]
        self.health = 100

    def draw(self, surface):
        # Base card
        pygame.draw.rect(surface, self.color, self.rect, 0, border_radius=5)
        # Icon/symbol
        self._draw_icon(surface)
        # Border
        pygame.draw.rect(surface, tuple(min(255, c+50) for c in self.color), self.rect, 2, border_radius=5)
    
    def activate(self, virus, env):
        # sfx: card_activate.wav
        return False # Returns True if it destroyed the virus

    def _draw_icon(self, surface):
        pass # Implemented by subclasses

class ReflectorCard(Card):
    def __init__(self, rect):
        super().__init__(rect, "REFLECTOR")

    def activate(self, virus, env):
        # Reflect based on which side of the card was hit
        if abs(virus.rect.centerx - self.rect.centerx) > abs(virus.rect.centery - self.rect.centery):
            virus.vel[0] *= -1.1 # Reflect with a small speed boost
        else:
            virus.vel[1] *= -1.1
        # sfx: reflect.wav
        env._create_particles(virus.pos, self.color, 5)
        return False

    def _draw_icon(self, surface):
        center = self.rect.center
        pygame.draw.lines(surface, GameEnv.COLOR_BG, False, [
            (center[0] - 10, center[1] - 15),
            (center[0] + 10, center[1]),
            (center[0] - 10, center[1] + 15)
        ], 4)

class DamagerCard(Card):
    def __init__(self, rect):
        super().__init__(rect, "DAMAGER")

    def activate(self, virus, env):
        damage = 15
        virus.health -= damage
        env._create_particles(virus.pos, self.color, 8)
        # sfx: damage.wav
        if virus.health <= 0:
            env.viruses.remove(virus)
            env._create_particles(virus.pos, GameEnv.COLOR_VIRUS, 30)
            # sfx: virus_destroy.wav
            return True
        return False

    def _draw_icon(self, surface):
        center = self.rect.center
        points = []
        for i in range(8):
            angle = i * (2 * math.pi / 8)
            r = 15 if i % 2 == 0 else 8
            points.append((center[0] + r * math.cos(angle), center[1] + r * math.sin(angle)))
        pygame.draw.polygon(surface, GameEnv.COLOR_BG, points)

class SlowerCard(Card):
    def __init__(self, rect):
        super().__init__(rect, "SLOWER")

    def activate(self, virus, env):
        virus.vel *= 0.5 # Halve the virus speed
        env._create_particles(virus.pos, self.color, 10)
        # sfx: slow.wav
        return False

    def _draw_icon(self, surface):
        center = self.rect.center
        pygame.draw.circle(surface, GameEnv.COLOR_BG, center, 15)
        pygame.draw.circle(surface, self.color, center, 10)
        pygame.draw.circle(surface, GameEnv.COLOR_BG, center, 5)

if __name__ == '__main__':
    # --- Manual Play Example ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # The main __init__ already sets the dummy driver, but for manual play,
    # we unset it to see the window.
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    pygame.display.set_caption("Firewall Defense")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n--- CONTROLS ---")
    print(GameEnv.user_guide)
    print("Q or Close Window: Quit")
    print("------------------\n")

    while running:
        # Default action is no-op
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        keys = pygame.key.get_pressed()
        
        # Map keys to MultiDiscrete action space
        if keys[pygame.K_1] or keys[pygame.K_LEFT]:
            action[0] = 1
        elif keys[pygame.K_2] or keys[pygame.K_UP]:
            action[0] = 2
        elif keys[pygame.K_3] or keys[pygame.K_DOWN]:
            action[0] = 3
        elif keys[pygame.K_4] or keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
            
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            running = False
            pygame.time.wait(2000) # Pause to see final screen

        clock.tick(30) # Run at 30 FPS

    env.close()
    pygame.quit()