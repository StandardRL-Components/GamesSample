import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player assembles a bioluminescent sea creature
    to fight a giant leviathan. The core mechanic involves placing parts with
    magnetic connectors to create chain reactions, dealing damage to the boss.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Assemble a bioluminescent sea creature to fight a giant leviathan. Place parts with "
        "magnetic connectors to create chain reactions and deal damage to the boss."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor. Press space to place a part and "
        "shift to discard the current part."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = (7, 5)
    GRID_CELL_SIZE = 48
    GRID_TOP_LEFT = (
        (SCREEN_WIDTH - GRID_SIZE[0] * GRID_CELL_SIZE) // 2,
        SCREEN_HEIGHT - GRID_SIZE[1] * GRID_CELL_SIZE - 20,
    )
    MAX_STEPS = 1000
    INITIAL_DECK_SIZE = 30
    LEVIATHAN_MAX_HEALTH = 100.0

    # --- Colors ---
    COLOR_BG = (10, 20, 40)
    COLOR_GRID = (30, 50, 80)
    COLOR_CURSOR = (255, 255, 100, 100)
    COLOR_TEXT = (220, 220, 255)
    COLOR_LEVIATHAN = (180, 40, 60)
    COLOR_LEVIATHAN_EYE = (255, 100, 100)
    COLOR_CHAIN_REACTION = (255, 255, 0)
    COLOR_HEALTH_HIGH = (80, 200, 80)
    COLOR_HEALTH_LOW = (200, 80, 80)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_damage = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        self._define_card_types()
        # State variables are initialized in reset()
        self.np_random = None

    def _define_card_types(self):
        """Defines the properties of each placeable creature part."""
        self.CARD_TYPES = {
            1: {"name": "I-Node", "color": (50, 150, 255), "connectors": {"N": 1, "E": 0, "S": 1, "W": 0}},
            2: {"name": "L-Node", "color": (50, 255, 150), "connectors": {"N": 1, "E": 1, "S": 0, "W": 0}},
            3: {"name": "T-Node", "color": (255, 150, 50), "connectors": {"N": 1, "E": 1, "S": 1, "W": 0}},
            4: {"name": "X-Node", "color": (200, 100, 255), "connectors": {"N": 1, "E": 1, "S": 1, "W": 1}},
            5: {"name": "End-Cap", "color": (150, 150, 150), "connectors": {"N": 1, "E": 0, "S": 0, "W": 0}},
        }
        self.CONNECTOR_SYMBOLS = {
            1: "●", # Circle for type 1
        }

    def _initialize_state(self):
        """Initializes all game state variables. Called by reset."""
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_status = ""

        self.leviathan_health = self.LEVIATHAN_MAX_HEALTH
        self.leviathan_regen_active = False
        self.leviathan_pulse = 0

        self.creature_grid = np.zeros(self.GRID_SIZE, dtype=int)
        self.cursor_pos = [self.GRID_SIZE[0] // 2, self.GRID_SIZE[1] // 2]
        
        card_keys = list(self.CARD_TYPES.keys())
        self.deck = self.np_random.choice(card_keys, size=self.INITIAL_DECK_SIZE).tolist()
        self.current_card = None
        self._draw_card()

        self.particles = []
        self.chain_effect_frames = 0

        self.prev_space_held = False
        self.prev_shift_held = False
        self.step_reward = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state()
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.step_reward = 0.0
        self.steps += 1
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        self._handle_input(movement, space_press, shift_press)
        self._update_game_state()

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated and self.win_status == "WIN":
            self.step_reward += 100.0
        elif terminated and self.win_status != "WIN":
            self.step_reward -= 100.0
        
        if truncated:
            self.game_over = True
            self.win_status = "LOSS (Timeout)"


        return (
            self._get_observation(),
            self.step_reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_press, shift_press):
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        if movement == 2: self.cursor_pos[1] += 1  # Down
        if movement == 3: self.cursor_pos[0] -= 1  # Left
        if movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE[0] - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE[1] - 1)

        if space_press:
            self._place_card()
        elif shift_press:
            self._discard_card()

    def _place_card(self):
        x, y = self.cursor_pos
        if self.creature_grid[x, y] == 0 and self.current_card is not None:
            # Place the card
            self.creature_grid[x, y] = self.current_card
            self.step_reward += 0.1
            # SFX: place_card.wav

            # Calculate chain reaction
            chain = self._calculate_chain((x, y))
            if len(chain) > 1:
                self.step_reward += 0.5
                damage = len(chain) ** 1.6 # Exponential damage scaling
                
                # Apply damage
                prev_health = self.leviathan_health
                self.leviathan_health -= damage
                actual_damage = prev_health - max(0, self.leviathan_health)
                self.step_reward += 1.0 * (actual_damage / 10.0)
                
                # Visual Effects
                self.chain_effect_frames = 15 # show chain lines for 15 frames
                self._create_damage_particle(actual_damage)
                # SFX: chain_reaction.wav, leviathan_hurt.wav
            
            self._draw_card()

    def _discard_card(self):
        if self.current_card is not None:
            # SFX: discard.wav
            self._draw_card()

    def _draw_card(self):
        if self.deck:
            self.current_card = self.deck.pop(0)
        else:
            self.current_card = None

    def _calculate_chain(self, start_pos):
        q = [start_pos]
        visited = {start_pos}
        chain = []

        while q:
            x, y = q.pop(0)
            chain.append((x, y))
            card_id = self.creature_grid[x, y]
            if card_id == 0: continue
            
            card_connectors = self.CARD_TYPES[card_id]["connectors"]
            
            # Check neighbors
            for direction, (dx, dy) in {"N": (0, -1), "S": (0, 1), "W": (-1, 0), "E": (1, 0)}.items():
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < self.GRID_SIZE[0] and 0 <= ny < self.GRID_SIZE[1] and (nx, ny) not in visited:
                    neighbor_id = self.creature_grid[nx, ny]
                    if neighbor_id == 0: continue

                    neighbor_connectors = self.CARD_TYPES[neighbor_id]["connectors"]
                    
                    # Get opposite direction
                    opposite = {"N": "S", "S": "N", "E": "W", "W": "E"}[direction]
                    
                    # Check for a valid connection
                    if card_connectors[direction] > 0 and card_connectors[direction] == neighbor_connectors[opposite]:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return chain

    def _update_game_state(self):
        # Leviathan state
        self.leviathan_pulse = (self.leviathan_pulse + 0.1) % (2 * math.pi)
        if self.leviathan_health <= self.LEVIATHAN_MAX_HEALTH / 2:
            self.leviathan_regen_active = True
        if self.leviathan_regen_active and self.steps % 50 == 0:
            self.leviathan_health = min(self.LEVIATHAN_MAX_HEALTH, self.leviathan_health + 1.0)

        # Particle state
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['y'] -= p['vy']
            p['life'] -= 1
        
        # Chain effect visual timer
        if self.chain_effect_frames > 0:
            self.chain_effect_frames -= 1

    def _check_termination(self):
        if self.leviathan_health <= 0:
            self.game_over = True
            self.win_status = "WIN"
            return True
        if self.current_card is None and not np.any(self.creature_grid > 0): # Can't place first card
            self.game_over = True
            self.win_status = "LOSS (No Cards)"
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_effects()
        self._render_leviathan()
        self._render_grid_and_creature()
        self._render_cursor_and_held_card()
        self._render_particles()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
            
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background_effects(self):
        # Slow-moving "plankton" particles
        for i in range(20):
            x = (hash(i * 10) + self.steps / 4) % self.SCREEN_WIDTH
            y = (hash(i * 20) + self.steps / 8) % self.SCREEN_HEIGHT
            pygame.gfxdraw.pixel(self.screen, int(x), int(y), (40, 60, 100))

    def _render_leviathan(self):
        health_ratio = max(0, self.leviathan_health / self.LEVIATHAN_MAX_HEALTH)
        pulse_size = 5 * math.sin(self.leviathan_pulse)
        
        # Body segments
        for i in range(5):
            size = 30 + i * 5 + pulse_size * (1 - i/5)
            x = self.SCREEN_WIDTH / 2 + (i - 2) * 40
            y = 100 - i * 10
            pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), int(size), self.COLOR_LEVIATHAN)
            pygame.gfxdraw.aacircle(self.screen, int(x), int(y), int(size), self.COLOR_LEVIATHAN)

        # Glowing Eye
        eye_color = self.COLOR_LEVIATHAN_EYE
        if self.leviathan_regen_active:
            eye_color = (255, 200, 100) # Brighter when regenerating
        pygame.gfxdraw.filled_circle(self.screen, self.SCREEN_WIDTH // 2, 80, int(15 + pulse_size/2), eye_color)
        pygame.gfxdraw.aacircle(self.screen, self.SCREEN_WIDTH // 2, 80, int(15 + pulse_size/2), eye_color)

    def _render_grid_and_creature(self):
        # Render grid lines
        for x in range(self.GRID_SIZE[0] + 1):
            start = (self.GRID_TOP_LEFT[0] + x * self.GRID_CELL_SIZE, self.GRID_TOP_LEFT[1])
            end = (self.GRID_TOP_LEFT[0] + x * self.GRID_CELL_SIZE, self.GRID_TOP_LEFT[1] + self.GRID_SIZE[1] * self.GRID_CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)
        for y in range(self.GRID_SIZE[1] + 1):
            start = (self.GRID_TOP_LEFT[0], self.GRID_TOP_LEFT[1] + y * self.GRID_CELL_SIZE)
            end = (self.GRID_TOP_LEFT[0] + self.GRID_SIZE[0] * self.GRID_CELL_SIZE, self.GRID_TOP_LEFT[1] + y * self.GRID_CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)
            
        # Render placed cards and chain effects
        if self.chain_effect_frames > 0:
            chain = self._calculate_chain(tuple(self.cursor_pos))
            # Draw chain lines
            for x, y in chain:
                for nx, ny in chain:
                    if abs(x-nx) + abs(y-ny) == 1: # is neighbor
                        p1_x = self.GRID_TOP_LEFT[0] + x * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE // 2
                        p1_y = self.GRID_TOP_LEFT[1] + y * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE // 2
                        p2_x = self.GRID_TOP_LEFT[0] + nx * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE // 2
                        p2_y = self.GRID_TOP_LEFT[1] + ny * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE // 2
                        pygame.draw.aaline(self.screen, self.COLOR_CHAIN_REACTION, (p1_x, p1_y), (p2_x, p2_y), 2)

        for x in range(self.GRID_SIZE[0]):
            for y in range(self.GRID_SIZE[1]):
                card_id = self.creature_grid[x, y]
                if card_id != 0:
                    self._render_card(card_id, (x, y), on_grid=True)

    def _render_card(self, card_id, pos, on_grid=False, alpha=255):
        card = self.CARD_TYPES[card_id]
        if on_grid:
            px = self.GRID_TOP_LEFT[0] + pos[0] * self.GRID_CELL_SIZE
            py = self.GRID_TOP_LEFT[1] + pos[1] * self.GRID_CELL_SIZE
        else:
            px, py = pos

        rect = pygame.Rect(px + 4, py + 4, self.GRID_CELL_SIZE - 8, self.GRID_CELL_SIZE - 8)
        pygame.draw.rect(self.screen, card["color"], rect, border_radius=4)
        
        # Render connectors
        center_x, center_y = rect.center
        for direction, conn_type in card["connectors"].items():
            if conn_type > 0:
                symbol = self.CONNECTOR_SYMBOLS.get(conn_type, "?")
                text = self.font_ui.render(symbol, True, self.COLOR_TEXT)
                if direction == 'N': text_pos = text.get_rect(centerx=center_x, top=rect.top)
                elif direction == 'S': text_pos = text.get_rect(centerx=center_x, bottom=rect.bottom)
                elif direction == 'W': text_pos = text.get_rect(centery=center_y, left=rect.left)
                elif direction == 'E': text_pos = text.get_rect(centery=center_y, right=rect.right)
                self.screen.blit(text, text_pos)

    def _render_cursor_and_held_card(self):
        # Render cursor
        cursor_rect = pygame.Rect(
            self.GRID_TOP_LEFT[0] + self.cursor_pos[0] * self.GRID_CELL_SIZE,
            self.GRID_TOP_LEFT[1] + self.cursor_pos[1] * self.GRID_CELL_SIZE,
            self.GRID_CELL_SIZE, self.GRID_CELL_SIZE
        )
        s = pygame.Surface((self.GRID_CELL_SIZE, self.GRID_CELL_SIZE), pygame.SRCALPHA)
        s.fill(self.COLOR_CURSOR)
        self.screen.blit(s, cursor_rect.topleft)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color'] + (alpha,)
            text_surf = self.font_damage.render(p['text'], True, self.COLOR_TEXT)
            text_surf.set_alpha(alpha)
            self.screen.blit(text_surf, (p['x'], p['y']))

    def _create_damage_particle(self, damage):
        self.particles.append({
            'x': self.SCREEN_WIDTH / 2 + random.uniform(-30, 30),
            'y': 100,
            'vy': 1.5,
            'text': f"{damage:.1f}",
            'life': 60,
            'max_life': 60,
            'color': (255, 200, 100)
        })

    def _render_ui(self):
        # Leviathan Health Bar
        health_ratio = max(0, self.leviathan_health / self.LEVIATHAN_MAX_HEALTH)
        bar_width = 300
        bar_height = 20
        bar_x = (self.SCREEN_WIDTH - bar_width) // 2
        bar_y = 20
        pygame.draw.rect(self.screen, (50, 0, 0), (bar_x, bar_y, bar_width, bar_height))
        health_color = (
            int(self.COLOR_HEALTH_LOW[0] + (self.COLOR_HEALTH_HIGH[0] - self.COLOR_HEALTH_LOW[0]) * health_ratio),
            int(self.COLOR_HEALTH_LOW[1] + (self.COLOR_HEALTH_HIGH[1] - self.COLOR_HEALTH_LOW[1]) * health_ratio),
            int(self.COLOR_HEALTH_LOW[2] + (self.COLOR_HEALTH_HIGH[2] - self.COLOR_HEALTH_LOW[2]) * health_ratio),
        )
        pygame.draw.rect(self.screen, health_color, (bar_x, bar_y, int(bar_width * health_ratio), bar_height))
        
        health_text = self.font_ui.render(f"LEVIATHAN HP: {self.leviathan_health:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (bar_x, bar_y + bar_height + 5))
        
        # Current Card Info
        ui_card_x, ui_card_y = 20, self.SCREEN_HEIGHT - 60
        pygame.draw.rect(self.screen, self.COLOR_GRID, (ui_card_x-5, ui_card_y-5, self.GRID_CELL_SIZE+10, self.GRID_CELL_SIZE+10), 2, 5)
        if self.current_card is not None:
            self._render_card(self.current_card, (ui_card_x, ui_card_y))
            card_name = self.CARD_TYPES[self.current_card]["name"]
            card_text = self.font_ui.render(f"Held: {card_name}", True, self.COLOR_TEXT)
            self.screen.blit(card_text, (ui_card_x + self.GRID_CELL_SIZE + 15, ui_card_y + 15))
        
        # Deck Info
        deck_text = self.font_ui.render(f"Parts Left: {len(self.deck)}", True, self.COLOR_TEXT)
        self.screen.blit(deck_text, (self.SCREEN_WIDTH - deck_text.get_width() - 20, self.SCREEN_HEIGHT - 40))

    def _render_game_over(self):
        s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        self.screen.blit(s, (0, 0))
        
        if self.win_status == "WIN":
            text = "LEVIATHAN DEFEATED"
            color = self.COLOR_CHAIN_REACTION
        else:
            text = "MISSION FAILED"
            color = self.COLOR_LEVIATHAN_EYE
            
        win_text_surf = self.font_game_over.render(text, True, color)
        win_text_rect = win_text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(win_text_surf, win_text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "leviathan_health": self.leviathan_health,
            "cards_left": len(self.deck),
            "win_status": self.win_status
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Example Usage ---
    env = GameEnv()
    
    # --- Manual Play ---
    # This loop allows a human to play the game.
    # Use Arrow Keys to move, Space to place, Left Shift to discard.
    
    obs, info = env.reset(seed=42)
    
    # Pygame setup for rendering
    pygame.display.set_caption("Leviathan Chain")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # --- Action Mapping for Human ---
        movement = 0 # none
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        # --- Step the Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Health: {info['leviathan_health']:.1f}")
        
        # --- Render the Observation ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Status: {info['win_status']}")
            pygame.time.wait(3000) # Wait 3 seconds before resetting
            obs, info = env.reset(seed=random.randint(0, 10000))

        clock.tick(30) # Run at 30 FPS

    env.close()