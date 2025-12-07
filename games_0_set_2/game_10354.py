import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:20:08.241267
# Source Brief: brief_00354.md
# Brief Index: 354
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    An expert-crafted Gymnasium environment simulating an ancient Greek Olympian
    competition. The agent must craft equipment and compete in various events
    to win the most medals. This environment prioritizes visual quality and
    satisfying game feel, adhering to a stylized Greek pottery aesthetic.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Compete as an ancient Greek Olympian. Craft equipment from materials like wood and bronze, "
        "then enter events like the javelin and sprint to win medals."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to navigate menus. Press SHIFT to select crafting materials "
        "and SPACE to confirm your choice or select an event."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    MAX_ROUNDS = 5
    MAX_STEPS_PER_EPISODE = 1000

    # --- Colors (Greek Pottery Palette) ---
    COLOR_BG = (210, 170, 140)  # Terracotta
    COLOR_FIGURE = (20, 20, 20)      # Black
    COLOR_ACCENT = (245, 235, 220) # Cream
    COLOR_SELECT = (255, 255, 0)   # Bright Yellow for selection
    COLOR_GOLD = (255, 215, 0)
    COLOR_SILVER = (192, 192, 192)
    COLOR_BRONZE = (205, 127, 50)
    
    # --- Materials and Stats ---
    MATERIALS = {
        "Wood": {"stat": "agility", "color": (139, 69, 19)},
        "Bronze": {"stat": "power", "color": (184, 115, 51)},
        "Linen": {"stat": "speed", "color": (240, 230, 140)},
    }
    
    # --- Events ---
    EVENTS = [
        {"name": "Javelin", "stat": "power"},
        {"name": "200m Sprint", "stat": "speed"},
        {"name": "Wrestling", "stat": "power"},
        {"name": "Long Jump", "stat": "agility"},
        {"name": "Discus", "stat": "power"},
        {"name": "400m Race", "stat": "speed"},
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_lg = pygame.font.Font(None, 48)
        self.font_md = pygame.font.Font(None, 32)
        self.font_sm = pygame.font.Font(None, 24)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_round = 0
        self.game_phase = "CRAFT" # CRAFT, SELECT_EVENT, PLAY_EVENT, AWARD_MEDALS, GAME_OVER
        self.player_medals = {"gold": 0, "silver": 0, "bronze": 0}
        self.opponent_medals = {"gold": 0, "silver": 0, "bronze": 0}
        self.available_materials = []
        self.selected_materials = []
        self.crafted_stats = {}
        self.crafting_cursor = 0
        self.event_cursor = (0, 0)
        self.selected_event = None
        self.event_results = []
        self.phase_timer = 0
        self.last_action = np.array([0, 0, 0])
        self.particles = []

        # --- Opponent Simulation ---
        self.opponents = [
            {"name": "Achilles", "stats": {"power": 5, "speed": 8, "agility": 6}},
            {"name": "Hector", "stats": {"power": 7, "speed": 5, "agility": 7}},
        ]
        
        # self.reset() is called by the wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_round = 1
        self.player_medals = {"gold": 0, "silver": 0, "bronze": 0}
        self.opponent_medals = {"gold": 0, "silver": 0, "bronze": 0}
        self.last_action = np.array([0, 0, 0])
        self.particles = []

        self._start_crafting_phase()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0
        
        movement, space_action, shift_action = action[0], action[1], action[2]
        space_pressed = space_action == 1 and self.last_action[1] == 0
        shift_pressed = shift_action == 1 and self.last_action[2] == 0

        # --- Phase-based Logic ---
        if self.game_phase == "CRAFT":
            reward += self._handle_crafting_phase(movement, space_pressed, shift_pressed)
        elif self.game_phase == "SELECT_EVENT":
            reward += self._handle_event_selection_phase(movement, space_pressed)
        elif self.game_phase == "PLAY_EVENT":
            self._handle_play_event_phase()
        elif self.game_phase == "AWARD_MEDALS":
            reward += self._handle_award_medals_phase()
        elif self.game_phase == "GAME_OVER":
            pass # No actions in game over state

        self.last_action = action
        
        # --- Termination ---
        terminated = self.game_phase == "GAME_OVER" or self.steps >= self.MAX_STEPS_PER_EPISODE
        if terminated and not self.game_over:
             reward += self._calculate_final_reward()
             self.game_over = True
             self.game_phase = "GAME_OVER"

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    # --- Phase Handlers ---
    def _start_crafting_phase(self):
        self.game_phase = "CRAFT"
        self.selected_materials = []
        self.crafted_stats = {}
        self.crafting_cursor = 0
        self.available_materials = random.sample(list(self.MATERIALS.keys()), 3)

    def _handle_crafting_phase(self, movement, space_pressed, shift_pressed):
        if shift_pressed and len(self.selected_materials) < 2:
            # Add material to selection
            material_name = self.available_materials[self.crafting_cursor]
            self.selected_materials.append(material_name)
            # SFX: crafting_add_item.wav
            self._create_particles(120 + self.crafting_cursor * 150, 200, self.MATERIALS[material_name]["color"])

        if movement == 3 and self.last_action[0] != 3: # Left
            self.crafting_cursor = (self.crafting_cursor - 1) % len(self.available_materials)
        if movement == 4 and self.last_action[0] != 4: # Right
            self.crafting_cursor = (self.crafting_cursor + 1) % len(self.available_materials)

        if space_pressed and self.selected_materials:
            # Confirm craft
            self.crafted_stats = {"power": 0, "speed": 0, "agility": 0}
            for mat_name in self.selected_materials:
                stat = self.MATERIALS[mat_name]["stat"]
                self.crafted_stats[stat] += 5 # Each material adds 5 points
            self.game_phase = "SELECT_EVENT"
            self.event_cursor = (0, 0)
            # SFX: craft_complete.wav
            return 1.0
        return 0.0

    def _handle_event_selection_phase(self, movement, space_pressed):
        rows, cols = 2, 3
        if self.last_action[0] != movement: # Debounce movement
            if movement == 1: # Up
                self.event_cursor = (max(0, self.event_cursor[0] - 1), self.event_cursor[1])
            elif movement == 2: # Down
                self.event_cursor = (min(rows - 1, self.event_cursor[0] + 1), self.event_cursor[1])
            elif movement == 3: # Left
                self.event_cursor = (self.event_cursor[0], max(0, self.event_cursor[1] - 1))
            elif movement == 4: # Right
                self.event_cursor = (self.event_cursor[0], min(cols - 1, self.event_cursor[1] + 1))
        
        if space_pressed:
            event_index = self.event_cursor[0] * cols + self.event_cursor[1]
            self.selected_event = self.EVENTS[event_index]
            self.game_phase = "PLAY_EVENT"
            self.phase_timer = 0
            # SFX: event_selected.wav
            return 0.1
        return 0.0

    def _handle_play_event_phase(self):
        self.phase_timer += 1
        if self.phase_timer > self.FPS * 2: # 2-second animation
            # Calculate scores
            stat = self.selected_event["stat"]
            player_score = self.crafted_stats.get(stat, 0) + self.np_random.uniform(1, 5)
            
            self.event_results = [{"name": "Player", "score": player_score}]
            for opp in self.opponents:
                opp_score = opp["stats"][stat] + self.np_random.uniform(1, 5)
                self.event_results.append({"name": opp["name"], "score": opp_score})
            
            self.event_results.sort(key=lambda x: x["score"], reverse=True)
            
            self.game_phase = "AWARD_MEDALS"
            self.phase_timer = 0
            # SFX: crowd_cheer.wav

    def _handle_award_medals_phase(self):
        self.phase_timer += 1
        reward = 0
        if self.phase_timer > self.FPS * 3: # 3-second display
            # Award medals and reward
            medal_rewards = {0: 5.0, 1: 3.0, 2: 1.0}
            medal_types = {0: "gold", 1: "silver", 2: "bronze"}

            for i, result in enumerate(self.event_results):
                if result["name"] == "Player":
                    if i in medal_rewards:
                        reward = medal_rewards[i]
                        self.player_medals[medal_types[i]] += 1
                else: # Opponent medal
                    if i in medal_types:
                        self.opponent_medals[medal_types[i]] += 1
            
            self.current_round += 1
            if self.current_round > self.MAX_ROUNDS:
                self.game_phase = "GAME_OVER"
            else:
                self._start_crafting_phase()
        return reward
    
    def _calculate_final_reward(self):
        player_score = self.player_medals["gold"] * 3 + self.player_medals["silver"] * 2 + self.player_medals["bronze"]
        opp_score = self.opponent_medals["gold"] * 3 + self.opponent_medals["silver"] * 2 + self.opponent_medals["bronze"]
        # Simplified: compare player total medals vs combined opponent total
        if player_score > opp_score:
            return 50.0 # Victory
        elif player_score == opp_score:
            return 20.0 # Draw
        return 0 # Loss

    # --- Rendering ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._draw_decorative_borders()
        
        # Update and draw particles
        self._update_particles()
        
        # Render based on phase
        if self.game_phase == "CRAFT":
            self._render_crafting_ui()
        elif self.game_phase == "SELECT_EVENT":
            self._render_event_selection_ui()
        elif self.game_phase == "PLAY_EVENT":
            self._render_play_event_animation()
        elif self.game_phase == "AWARD_MEDALS":
            self._render_award_medals_ui()
        elif self.game_phase == "GAME_OVER":
            self._render_game_over_ui()

        self._render_hud()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_text(self, text, font, color, center_pos):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect(center=center_pos)
        self.screen.blit(text_surf, text_rect)

    def _draw_decorative_borders(self):
        rect = pygame.Rect(10, 10, self.SCREEN_WIDTH - 20, self.SCREEN_HEIGHT - 20)
        pygame.draw.rect(self.screen, self.COLOR_FIGURE, rect, 3)
        for i in range(15, self.SCREEN_WIDTH - 15, 15):
            pygame.draw.line(self.screen, self.COLOR_FIGURE, (i, 10), (i+5, 13), 2)
            pygame.draw.line(self.screen, self.COLOR_FIGURE, (i, rect.bottom), (i+5, rect.bottom-3), 2)
        for i in range(15, self.SCREEN_HEIGHT - 15, 15):
             pygame.draw.line(self.screen, self.COLOR_FIGURE, (10, i), (13, i+5), 2)
             pygame.draw.line(self.screen, self.COLOR_FIGURE, (rect.right, i), (rect.right-3, i+5), 2)
    
    def _render_hud(self):
        # Round counter
        self._render_text(f"Round: {self.current_round}/{self.MAX_ROUNDS}", self.font_md, self.COLOR_FIGURE, (self.SCREEN_WIDTH // 2, 35))
        
        # Player Medals
        self._render_text("You", self.font_md, self.COLOR_FIGURE, (100, 35))
        self._draw_medal_counts(self.player_medals, 100, 65)

        # Opponent Medals
        self._render_text("Rivals", self.font_md, self.COLOR_FIGURE, (self.SCREEN_WIDTH - 100, 35))
        self._draw_medal_counts(self.opponent_medals, self.SCREEN_WIDTH - 100, 65)

    def _draw_medal_counts(self, medals, cx, cy):
        pygame.gfxdraw.filled_circle(self.screen, cx - 30, cy, 10, self.COLOR_GOLD)
        self._render_text(str(medals["gold"]), self.font_sm, self.COLOR_FIGURE, (cx-30, cy))
        pygame.gfxdraw.filled_circle(self.screen, cx, cy, 10, self.COLOR_SILVER)
        self._render_text(str(medals["silver"]), self.font_sm, self.COLOR_FIGURE, (cx, cy))
        pygame.gfxdraw.filled_circle(self.screen, cx + 30, cy, 10, self.COLOR_BRONZE)
        self._render_text(str(medals["bronze"]), self.font_sm, self.COLOR_FIGURE, (cx+30, cy))

    def _render_crafting_ui(self):
        self._render_text("Craft Your Equipment", self.font_lg, self.COLOR_FIGURE, (self.SCREEN_WIDTH // 2, 120))
        
        # Display available materials
        for i, mat_name in enumerate(self.available_materials):
            x = 120 + i * 150
            y = 200
            color = self.MATERIALS[mat_name]["color"]
            pygame.draw.rect(self.screen, color, (x - 40, y - 40, 80, 80))
            pygame.draw.rect(self.screen, self.COLOR_FIGURE, (x - 40, y - 40, 80, 80), 3)
            self._render_text(mat_name, self.font_sm, self.COLOR_FIGURE, (x, y + 55))
            if i == self.crafting_cursor:
                pygame.draw.rect(self.screen, self.COLOR_SELECT, (x - 45, y - 45, 90, 90), 4)

        # Display selected materials
        self._render_text("Selected:", self.font_sm, self.COLOR_FIGURE, (self.SCREEN_WIDTH // 2 - 80, 320))
        for i, mat_name in enumerate(self.selected_materials):
            x = self.SCREEN_WIDTH // 2 + i * 50
            y = 320
            color = self.MATERIALS[mat_name]["color"]
            pygame.draw.rect(self.screen, color, (x-15, y-15, 30, 30))
            pygame.draw.rect(self.screen, self.COLOR_FIGURE, (x-15, y-15, 30, 30), 2)
            
        self._render_text("Use SHIFT to select, SPACE to craft", self.font_sm, self.COLOR_ACCENT, (self.SCREEN_WIDTH // 2, 370))

    def _render_event_selection_ui(self):
        self._render_text("Select an Event", self.font_lg, self.COLOR_FIGURE, (self.SCREEN_WIDTH // 2, 120))
        rows, cols = 2, 3
        for i, event in enumerate(self.EVENTS):
            row, col = divmod(i, cols)
            x = 160 + col * 160
            y = 180 + row * 80
            rect = pygame.Rect(x-70, y-30, 140, 60)
            pygame.draw.rect(self.screen, self.COLOR_ACCENT, rect, border_radius=5)
            pygame.draw.rect(self.screen, self.COLOR_FIGURE, rect, 2, border_radius=5)
            self._render_text(event["name"], self.font_sm, self.COLOR_FIGURE, rect.center)
            
            if (row, col) == self.event_cursor:
                 pygame.draw.rect(self.screen, self.COLOR_SELECT, rect, 4, border_radius=5)

    def _render_play_event_animation(self):
        self._render_text(self.selected_event["name"], self.font_lg, self.COLOR_FIGURE, (self.SCREEN_WIDTH // 2, 120))
        progress = self.phase_timer / (self.FPS * 2)
        
        # Simple generic animation: a moving projectile
        start_x, y = 80, 250
        end_x = self.SCREEN_WIDTH - 80
        current_x = start_x + (end_x - start_x) * progress
        height = math.sin(progress * math.pi) * 100
        
        # Draw "javelin"
        pygame.draw.line(self.screen, self.COLOR_FIGURE, (int(current_x - 20), int(y - height)), (int(current_x), int(y - height)), 4)
        # Draw ground
        pygame.draw.line(self.screen, self.COLOR_FIGURE, (50, y+10), (self.SCREEN_WIDTH - 50, y+10), 3)

    def _render_award_medals_ui(self):
        self._render_text("Results", self.font_lg, self.COLOR_FIGURE, (self.SCREEN_WIDTH // 2, 120))
        podium_colors = [self.COLOR_GOLD, self.COLOR_SILVER, self.COLOR_BRONZE]
        for i, result in enumerate(self.event_results):
            y = 180 + i * 50
            # Medal
            if i < 3:
                pygame.gfxdraw.filled_circle(self.screen, 150, y, 15, podium_colors[i])
            # Name and Score
            self._render_text(f"{i+1}. {result['name']}", self.font_md, self.COLOR_FIGURE, (300, y))
            self._render_text(f"Score: {result['score']:.1f}", self.font_md, self.COLOR_FIGURE, (500, y))

    def _render_game_over_ui(self):
        player_score = self.player_medals["gold"] * 3 + self.player_medals["silver"] * 2 + self.player_medals["bronze"]
        opp_score = self.opponent_medals["gold"] * 3 + self.opponent_medals["silver"] * 2 + self.opponent_medals["bronze"]
        
        if player_score > opp_score:
            msg = "Victory!"
            color = self.COLOR_GOLD
        elif player_score == opp_score:
            msg = "A Heroic Draw!"
            color = self.COLOR_SILVER
        else:
            msg = "Defeat..."
            color = self.COLOR_FIGURE
            
        self._render_text("Game Over", self.font_lg, self.COLOR_FIGURE, (self.SCREEN_WIDTH // 2, 150))
        self._render_text(msg, self.font_lg, color, (self.SCREEN_WIDTH // 2, 220))
        self._render_text("Call reset() to play again", self.font_sm, self.COLOR_FIGURE, (self.SCREEN_WIDTH // 2, 300))

    # --- Particles for Visual Polish ---
    def _create_particles(self, x, y, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            particle = {
                "pos": [x, y],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(20, 40),
                "color": color
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1  # Gravity
            p["life"] -= 1
        
        self.particles = [p for p in self.particles if p["life"] > 0]
        
        for p in self.particles:
            size = max(0, int(p["life"] / 8))
            pygame.draw.circle(self.screen, p["color"], (int(p["pos"][0]), int(p["pos"][1])), size)

    # --- Gymnasium Compliance ---
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "round": self.current_round,
            "phase": self.game_phase,
            "player_medals": self.player_medals,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Example Usage and Manual Play ---
    # The main-guard is useful for local testing and debugging.
    # It will not be run by the evaluation server.
    
    # To test with rendering, unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window for human play
    pygame.display.set_caption("Olympian Craft")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("--------------------")

    # The main game loop
    while not done:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([movement, space_held, shift_held])
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if reward != 0:
            print(f"Round: {info['round']}, Phase: {info['phase']}, Reward: {reward:.2f}, Score: {info['score']:.2f}")

        # Render to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Event handling for quitting
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        clock.tick(GameEnv.FPS)

    print("\nGame Over!")
    print(f"Final Info: {info}")
    env.close()