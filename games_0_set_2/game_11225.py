import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:42:07.247420
# Source Brief: brief_01225.md
# Brief Index: 1225
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent arranges physics-based flower cards.

    The agent controls a cursor to select different types of flower cards and then
    "teleports" a petal into the play area. This petal acts as a projectile,
    colliding with and imparting force to the cards. Each card type has unique
    physics properties (heavy, buoyant, magnetic). The goal is to arrange the
    cards into a target pattern, causing flowers to "bloom" visually.

    The environment prioritizes visual quality and satisfying "game feel", with
    smooth animations, particle effects, and clear visual feedback for all actions.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Arrange physics-based flower cards into a target pattern. Select cards and fire petals to move them into place, causing them to bloom."
    )
    user_guide = (
        "Use arrow keys to move the cursor. Press space to select a card type or fire a petal. Press shift to reset all placed cards."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 2000

    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_GRID = (30, 40, 60)
    COLOR_UI_BG = (10, 15, 25)
    COLOR_UI_BORDER = (40, 50, 70)
    COLOR_TEXT = (220, 220, 240)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_AIM_LINE = (255, 255, 255, 100)

    # Card Types & Properties
    CARD_TYPES = {
        "STONE": {"color": (160, 160, 180), "mass": 2.0, "gravity_mod": 1.0, "buoyancy": 0, "magnetism": 0},
        "BUBBLE": {"color": (100, 180, 255), "mass": 0.5, "gravity_mod": 0.0, "buoyancy": 1.5, "magnetism": 0},
        "POLLEN": {"color": (255, 220, 80), "mass": 1.0, "gravity_mod": 0.1, "buoyancy": 0, "magnetism": 200},
    }
    CARD_RADIUS = 15

    # Physics
    GRAVITY = 9.8
    DAMPING = 0.98
    PETAL_SPEED = 300
    PETAL_RADIUS = 5
    PETAL_MASS = 0.5

    # Game Areas
    PLAY_AREA = pygame.Rect(10, 10, WIDTH - 20, HEIGHT - 80)
    HAND_AREA = pygame.Rect(0, HEIGHT - 70, WIDTH, 70)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # --- Game State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.reward_this_step = 0
        self.game_over = False
        self.pollen_resource = 0
        self.max_pollen = 100
        self.cursor_pos = np.array([0.0, 0.0])
        self.game_phase = "SELECT"  # 'SELECT' or 'AIM'
        self.selected_card_type = None
        self.placed_cards = []
        self.petals = []
        self.particles = []
        self.target_pattern = []
        self.completed_targets = []
        self.hand_card_positions = {}
        self.previous_space_state = 0
        self.previous_shift_state = 0
        self.difficulty = 3 # Initial number of targets

        self.reset()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.reward_this_step = 0
        self.game_over = False

        self.pollen_resource = self.max_pollen
        self.cursor_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        self.game_phase = "SELECT"
        self.selected_card_type = None

        self.placed_cards = []
        self.petals = []
        self.particles = []
        self.completed_targets = []
        
        # In a real game, difficulty might persist. For this env, we reset it.
        if options and "difficulty" in options:
            self.difficulty = options["difficulty"]

        self._generate_target_pattern(self.difficulty)

        # Set up static positions for cards in the hand
        card_keys = list(self.CARD_TYPES.keys())
        for i, key in enumerate(card_keys):
            x = self.WIDTH / 2 + (i - (len(card_keys) - 1) / 2) * 100
            y = self.HAND_AREA.centery
            self.hand_card_positions[key] = np.array([x, y])

        self.previous_space_state = 0
        self.previous_shift_state = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = 0
        self.game_over = False

        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1], action[2]
        space_press = space_held and not self.previous_space_state
        shift_press = shift_held and not self.previous_shift_state
        self.previous_space_state = space_held
        self.previous_shift_state = shift_held

        self._handle_input(movement, space_press, shift_press)

        # --- Update Game Logic ---
        self._update_physics(1.0 / self.FPS)
        self._update_petals()
        self._update_particles()
        self._check_pattern_completion()

        self.steps += 1

        # --- Calculate Reward & Termination ---
        self.score += self.reward_this_step
        terminated = self._check_termination()

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,  # truncated
            self._get_info()
        )

    def _handle_input(self, movement, space_press, shift_press):
        # --- Cursor Movement ---
        cursor_speed = 10
        if movement == 1: self.cursor_pos[1] -= cursor_speed
        elif movement == 2: self.cursor_pos[1] += cursor_speed
        elif movement == 3: self.cursor_pos[0] -= cursor_speed
        elif movement == 4: self.cursor_pos[0] += cursor_speed
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)

        # --- Action Button (Space) ---
        if space_press:
            if self.game_phase == "SELECT":
                for card_type, pos in self.hand_card_positions.items():
                    if np.linalg.norm(self.cursor_pos - pos) < self.CARD_RADIUS + 5:
                        self.selected_card_type = card_type
                        self.game_phase = "AIM"
                        self._create_particles(pos, self.CARD_TYPES[card_type]["color"], 10, 2) # Sound: UI_Select
                        break
            elif self.game_phase == "AIM" and self.PLAY_AREA.collidepoint(self.cursor_pos):
                self._fire_petal()
                self.game_phase = "SELECT"
                self.selected_card_type = None

        # --- Reset Button (Shift) ---
        if shift_press:
            self.reward_this_step -= 10
            # Remove all placed cards and reset their state
            for card in self.placed_cards:
                 self._create_particles(card['pos'], card['color'], 15, 3) # Sound: Reset_Poof
            self.placed_cards = []
            self.completed_targets = [] # Also un-complete targets
            self.pollen_resource = max(0, self.pollen_resource - 20) # Cost for resetting
            

    def _fire_petal(self):
        pollen_cost = 5
        if self.pollen_resource >= pollen_cost:
            self.pollen_resource -= pollen_cost
            self.reward_this_step -= 0.1 * pollen_cost # Resource consumption penalty
            self.reward_this_step += 1 # Reward for taking an action

            origin_pos = self.hand_card_positions[self.selected_card_type]
            target_pos = self.cursor_pos
            direction = target_pos - origin_pos
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)

            # A petal is "teleported" to the cursor and then fired
            new_petal = {
                "pos": target_pos.copy(),
                "vel": direction * self.PETAL_SPEED,
                "color": self.CARD_TYPES[self.selected_card_type]["color"],
                "card_type": self.selected_card_type,
            }
            self.petals.append(new_petal)
            self._create_particles(target_pos, new_petal["color"], 20, 3) # Sound: Petal_Fire
        else:
            # Not enough pollen, maybe a "fail" sound/visual
            self._create_particles(self.cursor_pos, (100, 100, 100), 5, 1) # Sound: UI_Fail

    def _update_physics(self, dt):
        # --- Update Cards ---
        for i, card in enumerate(self.placed_cards):
            props = self.CARD_TYPES[card["type"]]
            
            # Forces
            force = np.array([0.0, 0.0])
            force[1] += props["gravity_mod"] * self.GRAVITY * props["mass"] # Gravity
            force[1] -= props["buoyancy"] * props["mass"] # Buoyancy

            # Magnetism (Pollen cards attract each other)
            if props["magnetism"] > 0:
                for other_card in self.placed_cards:
                    if card is other_card and other_card["type"] == card["type"]:
                        vec = other_card["pos"] - card["pos"]
                        dist_sq = max(1, vec.dot(vec))
                        if dist_sq < (self.CARD_RADIUS * 8)**2:
                            force_mag = props["magnetism"] * props["mass"] / dist_sq
                            force += (vec / math.sqrt(dist_sq)) * force_mag

            # Update velocity and position
            card["vel"] += (force / props["mass"]) * dt
            card["vel"] *= self.DAMPING
            card["pos"] += card["vel"] * dt

            # Wall collisions
            if not self.PLAY_AREA.collidepoint(card["pos"]):
                if card["pos"][0] < self.PLAY_AREA.left + self.CARD_RADIUS or card["pos"][0] > self.PLAY_AREA.right - self.CARD_RADIUS:
                    card["vel"][0] *= -0.8
                if card["pos"][1] < self.PLAY_AREA.top + self.CARD_RADIUS or card["pos"][1] > self.PLAY_AREA.bottom - self.CARD_RADIUS:
                    card["vel"][1] *= -0.8
                card["pos"][0] = np.clip(card["pos"][0], self.PLAY_AREA.left + self.CARD_RADIUS, self.PLAY_AREA.right - self.CARD_RADIUS)
                card["pos"][1] = np.clip(card["pos"][1], self.PLAY_AREA.top + self.CARD_RADIUS, self.PLAY_AREA.bottom - self.CARD_RADIUS)

        # --- Card-Card Collisions ---
        for i in range(len(self.placed_cards)):
            for j in range(i + 1, len(self.placed_cards)):
                c1, c2 = self.placed_cards[i], self.placed_cards[j]
                vec = c2["pos"] - c1["pos"]
                dist = np.linalg.norm(vec)
                min_dist = self.CARD_RADIUS * 2
                if dist < min_dist:
                    # Resolve overlap
                    overlap = min_dist - dist
                    c1["pos"] -= (vec / dist) * overlap / 2
                    c2["pos"] += (vec / dist) * overlap / 2

                    # Elastic collision response
                    m1, m2 = self.CARD_TYPES[c1["type"]]["mass"], self.CARD_TYPES[c2["type"]]["mass"]
                    v1, v2 = c1["vel"], c2["vel"]
                    
                    normal = vec / dist
                    p = 2 * (np.dot(v1, normal) - np.dot(v2, normal)) / (m1 + m2)
                    c1["vel"] -= p * m2 * normal
                    c2["vel"] += p * m1 * normal
                    self._create_particles((c1['pos'] + c2['pos'])/2, (200,200,200), 5, 1) # Sound: Card_Clink


    def _update_petals(self):
        for petal in self.petals[:]:
            petal["pos"] += petal["vel"] * (1.0 / self.FPS)
            
            # Petal-Card collision
            collided = False
            for card in self.placed_cards:
                if np.linalg.norm(petal["pos"] - card["pos"]) < self.CARD_RADIUS + self.PETAL_RADIUS:
                    # Impart force
                    impulse = petal["vel"] * self.PETAL_MASS
                    card["vel"] += impulse / self.CARD_TYPES[card["type"]]["mass"]
                    
                    self._create_particles(petal["pos"], petal["color"], 25, 4) # Sound: Petal_Impact
                    self.petals.remove(petal)
                    collided = True
                    break
            if collided: continue

            # If petal leaves play area, or if a new card is placed, remove it
            if not self.PLAY_AREA.contains(pygame.Rect(petal["pos"][0], petal["pos"][1], 1, 1)):
                if not self._place_new_card(petal):
                     self._create_particles(petal['pos'], (100,100,100), 10, 1) # Sound: Petal_Fizzle
                self.petals.remove(petal)

    def _place_new_card(self, petal):
        # Place a new card where the petal landed, if inside play area
        pos = petal["pos"]
        if self.PLAY_AREA.collidepoint(pos):
            self.placed_cards.append({
                "pos": pos,
                "vel": np.array([0.0, 0.0]),
                "type": petal["card_type"],
                "color": petal["color"],
                "time_in_target": 0
            })
            self._create_particles(pos, petal["color"], 30, 5) # Sound: Card_Place
            return True
        return False

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            p["radius"] = max(0, p["radius"] * 0.95)
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _check_pattern_completion(self):
        settle_time_req = 30 # frames (1 second at 30fps)
        
        for i, target in enumerate(self.target_pattern):
            if i in self.completed_targets:
                continue

            target_pos = target["pos"]
            target_type = target["type"]
            found_card = False

            for card in self.placed_cards:
                if card["type"] == target_type:
                    dist = np.linalg.norm(card["pos"] - target_pos)
                    if dist < self.CARD_RADIUS and np.linalg.norm(card["vel"]) < 1.0:
                        card["time_in_target"] += 1
                        if card["time_in_target"] >= settle_time_req:
                            self.completed_targets.append(i)
                            self.score += 50 # Base score for one flower
                            self.reward_this_step += 5 # Event reward for growth
                            self.pollen_resource = min(self.max_pollen, self.pollen_resource + 25) # Resource bonus
                            self._create_particles(target_pos, self.CARD_TYPES[target_type]["color"], 50, 10, is_flower=True) # Sound: Success_Chime
                            found_card = True
                            break # Only one card can complete a target
                    else:
                        card["time_in_target"] = 0 # Reset if it moves out or moves too fast
            if found_card:
                # To avoid re-checking this target
                continue

    def _check_termination(self):
        if self.game_over: return True
        
        # Win condition
        if len(self.completed_targets) == len(self.target_pattern):
            self.reward_this_step += 50  # Big bonus for completing the whole display
            self.game_over = True
            return True
        
        # Lose condition
        if self.pollen_resource <= 0 and not self.petals:
            self.reward_this_step -= 10 # Penalty for running out of resources
            self.game_over = True
            return True
            
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
            
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "pollen": self.pollen_resource,
            "phase": self.game_phase,
            "targets_completed": len(self.completed_targets),
            "targets_total": len(self.target_pattern)
        }
    
    def _render_game(self):
        # --- Draw Grid ---
        for i in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

        # --- Draw Target Pattern ---
        for i, target in enumerate(self.target_pattern):
            if i not in self.completed_targets:
                self._draw_glow(self.screen, (*self.CARD_TYPES[target["type"]]["color"], 50), target["pos"], self.CARD_RADIUS + 2, 3)
                pygame.gfxdraw.aacircle(self.screen, int(target["pos"][0]), int(target["pos"][1]), self.CARD_RADIUS, (*self.CARD_TYPES[target["type"]]["color"], 100))

        # --- Draw Placed Cards & Completed Flowers ---
        for card in self.placed_cards:
            self._draw_card(self.screen, card)
        
        for i in self.completed_targets:
            target = self.target_pattern[i]
            self._draw_flower(self.screen, target['pos'], self.CARD_TYPES[target['type']]['color'])

        # --- Draw Petals & Particles ---
        for petal in self.petals:
            pygame.gfxdraw.filled_circle(self.screen, int(petal["pos"][0]), int(petal["pos"][1]), self.PETAL_RADIUS, petal["color"])
            pygame.gfxdraw.aacircle(self.screen, int(petal["pos"][0]), int(petal["pos"][1]), self.PETAL_RADIUS, (255,255,255))
            
        for p in self.particles:
            alpha_color = (*p["color"], max(0, min(255, int(255 * (p["lifespan"] / p["max_lifespan"])))) )
            self._draw_glow(self.screen, alpha_color, p['pos'], int(p['radius']), 2)


    def _render_ui(self):
        # --- UI Background ---
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, self.HAND_AREA)
        pygame.draw.line(self.screen, self.COLOR_UI_BORDER, self.HAND_AREA.topleft, self.HAND_AREA.topright, 2)

        # --- Pollen Bar ---
        pollen_rect = pygame.Rect(10, self.HAND_AREA.y + 10, 150, 20)
        pygame.draw.rect(self.screen, self.COLOR_UI_BORDER, pollen_rect, 1)
        fill_width = max(0, (self.pollen_resource / self.max_pollen) * (pollen_rect.width - 4))
        fill_rect = pygame.Rect(pollen_rect.x + 2, pollen_rect.y + 2, fill_width, pollen_rect.height - 4)
        pygame.draw.rect(self.screen, self.CARD_TYPES["POLLEN"]["color"], fill_rect)
        pollen_text = self.font_small.render("POLLEN", True, self.COLOR_TEXT)
        self.screen.blit(pollen_text, (pollen_rect.x, pollen_rect.bottom + 5))

        # --- Hand Cards ---
        for card_type, pos in self.hand_card_positions.items():
            card_data = {"pos": pos, "type": card_type, "color": self.CARD_TYPES[card_type]["color"]}
            is_selected = (self.selected_card_type == card_type)
            is_hovered = (self.game_phase == "SELECT" and np.linalg.norm(self.cursor_pos - pos) < self.CARD_RADIUS + 5)
            self._draw_card(self.screen, card_data, is_selected=is_selected, is_hovered=is_hovered)

        # --- Score and Step Text ---
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 15, 10))
        
        # --- Cursor and Aiming Line ---
        if self.game_phase == "AIM":
            # Aiming reticle
            pygame.draw.line(self.screen, self.COLOR_CURSOR, (self.cursor_pos[0] - 10, self.cursor_pos[1]), (self.cursor_pos[0] + 10, self.cursor_pos[1]), 2)
            pygame.draw.line(self.screen, self.COLOR_CURSOR, (self.cursor_pos[0], self.cursor_pos[1] - 10), (self.cursor_pos[0], self.cursor_pos[1] + 10), 2)
            pygame.gfxdraw.aacircle(self.screen, int(self.cursor_pos[0]), int(self.cursor_pos[1]), 8, self.COLOR_CURSOR)
            # Aiming line from hand
            origin_pos = self.hand_card_positions[self.selected_card_type]
            pygame.draw.aaline(self.screen, self.COLOR_AIM_LINE, origin_pos, self.cursor_pos, 2)
        else:
            # Default cursor
            pygame.gfxdraw.filled_circle(self.screen, int(self.cursor_pos[0]), int(self.cursor_pos[1]), 4, self.COLOR_CURSOR)
            pygame.gfxdraw.aacircle(self.screen, int(self.cursor_pos[0]), int(self.cursor_pos[1]), 7, self.COLOR_CURSOR)

    def _draw_card(self, surface, card, is_selected=False, is_hovered=False):
        pos = (int(card["pos"][0]), int(card["pos"][1]))
        color = card["color"]
        
        if is_selected:
            self._draw_glow(surface, (*color, 150), card["pos"], self.CARD_RADIUS + 5, 5)
        elif is_hovered:
            self._draw_glow(surface, (255, 255, 255, 100), card["pos"], self.CARD_RADIUS + 3, 3)
        
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], self.CARD_RADIUS, color)
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], self.CARD_RADIUS, (255,255,255,50))
        # Add a small highlight
        highlight_pos = (pos[0] + 5, pos[1] - 5)
        pygame.gfxdraw.filled_circle(surface, highlight_pos[0], highlight_pos[1], 4, (255,255,255,80))

    def _draw_flower(self, surface, pos, color):
        self._draw_glow(surface, (*color, 100), pos, self.CARD_RADIUS * 1.5, 5)
        for i in range(6):
            angle = i * (2 * math.pi / 6) + (self.steps / 20.0) # Slow rotation
            petal_pos = (
                int(pos[0] + math.cos(angle) * self.CARD_RADIUS * 0.8),
                int(pos[1] + math.sin(angle) * self.CARD_RADIUS * 0.8)
            )
            pygame.gfxdraw.filled_circle(surface, petal_pos[0], petal_pos[1], self.CARD_RADIUS // 2, color)
            pygame.gfxdraw.aacircle(surface, petal_pos[0], petal_pos[1], self.CARD_RADIUS // 2, (255,255,255,100))
        pygame.gfxdraw.filled_circle(surface, int(pos[0]), int(pos[1]), self.CARD_RADIUS // 2, (255,255,150))


    def _draw_glow(self, surface, color, pos, radius, steps):
        pos_int = (int(pos[0]), int(pos[1]))
        for i in range(steps, 0, -1):
            alpha = color[3] * (i / steps)
            rad = radius * (i / steps)
            if rad > 0:
                s = pygame.Surface((rad*2, rad*2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*color[:3], int(alpha / steps)), (rad, rad), rad)
                surface.blit(s, (pos_int[0] - rad, pos_int[1] - rad))

    def _create_particles(self, pos, color, count, speed_mult, is_flower=False):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            lifespan = self.np_random.integers(20, 40)
            radius = self.np_random.uniform(2, 6) if not is_flower else self.np_random.uniform(5, 10)
            self.particles.append({
                "pos": pos + self.np_random.uniform(-5, 5, size=2),
                "vel": np.array([math.cos(angle), math.sin(angle)]) * speed,
                "color": color,
                "radius": radius,
                "lifespan": lifespan,
                "max_lifespan": lifespan
            })

    def _generate_target_pattern(self, num_targets):
        self.target_pattern = []
        card_types = list(self.CARD_TYPES.keys())
        for _ in range(num_targets):
            # Ensure targets don't overlap too much
            while True:
                pos = np.array([
                    self.np_random.uniform(self.PLAY_AREA.left + 30, self.PLAY_AREA.right - 30),
                    self.np_random.uniform(self.PLAY_AREA.top + 30, self.PLAY_AREA.bottom - 30)
                ])
                if all(np.linalg.norm(pos - t["pos"]) > self.CARD_RADIUS * 3 for t in self.target_pattern):
                    break
            
            self.target_pattern.append({
                "pos": pos,
                "type": self.np_random.choice(card_types)
            })

    def close(self):
        pygame.quit()

# --- Example Usage ---
if __name__ == "__main__":
    # The main loop is for human play and requires a display.
    # Unset the dummy video driver if you want to run this interactively.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    pygame.display.set_caption("Floral Arrangement AI")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    print(GameEnv.game_description)
    print(GameEnv.user_guide)
    print("  - Q: Quit")

    while not done:
        # --- Human Input Mapping ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Pygame Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                done = True
        
        clock.tick(GameEnv.FPS)

    print(f"Game Over. Final Score: {info['score']}")
    env.close()