import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:27:27.549385
# Source Brief: brief_02179.md
# Brief Index: 2179
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where a bee collects pollen from flowers.

    The bee navigates a 2D field, collecting nectar from flowers. Each unit of
    nectar allows the bee to draw a random ability card. These cards provide
    temporary boosts, such as increased speed or a larger pollen collection radius.
    The goal is to maximize the pollen collected within a fixed time limit.

    **Visuals:**
    - The game features a vibrant, stylized aesthetic with smooth animations.
    - The bee has flapping wings and a glow effect.
    - Flowers change color to indicate their nectar status.
    - Particle effects provide feedback for collecting pollen and using abilities.
    - A clear UI displays the score, timer, nectar count, and card hand.

    **Gameplay:**
    - The bee's movement is physics-based, with acceleration and friction.
    - The `MultiDiscrete([5, 2, 2])` action space allows simultaneous
      movement and ability usage (drawing/playing cards).
    - The game ends when the timer reaches zero.

    **Rewards:**
    - The agent is rewarded for collecting nectar and pollen, drawing cards,
      and reaching score milestones.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A bee collects pollen from flowers to score points. Collect nectar to draw ability cards that provide temporary boosts like increased speed or a larger collection radius."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the bee. Press space to draw an ability card and shift to use the first card in your hand."
    )
    auto_advance = True


    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1800  # Approx 30 seconds at 60 FPS

    # Colors
    COLOR_BG = (135, 206, 235)  # Sky Blue
    COLOR_GRASS = (50, 150, 50)
    COLOR_UI_BG = (0, 0, 0, 128)
    COLOR_TEXT = (255, 255, 255)
    COLOR_BEE = (255, 223, 0)  # Gold
    COLOR_BEE_GLOW = (255, 223, 0, 60)
    COLOR_BEE_WING = (255, 255, 255, 180)
    COLOR_FLOWER_FULL = (255, 20, 147)  # Deep Pink
    COLOR_FLOWER_EMPTY = (105, 10, 67)
    COLOR_FLOWER_CENTER = (255, 255, 0)
    COLOR_POLLEN = (255, 255, 102) # Light Yellow
    COLOR_TIMER_WARN = (255, 100, 100)

    # Game Parameters
    BEE_ACCELERATION = 0.6
    BEE_FRICTION = 0.96
    BEE_MAX_SPEED = 10.0
    BEE_RADIUS = 12
    FLOWER_COUNT = 10
    FLOWER_RADIUS = 15
    FLOWER_RESPAWN_TIME = 300  # in steps
    POLLEN_PER_FLOWER = 10
    CARD_HAND_LIMIT = 3

    CARDS = {
        "SPEED": {"name": "Haste", "duration": 240, "color": (100, 100, 255)},
        "MAGNET": {"name": "Attract", "duration": 300, "color": (255, 100, 255)},
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
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # Initialize state variables (defined in reset)
        self.bee_pos = None
        self.bee_vel = None
        self.flowers = []
        self.pollen_particles = []
        self.floating_texts = []
        self.card_hand = []
        self.active_effects = {}
        self.steps = 0
        self.score = 0
        self.nectar_bank = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.last_score_milestone = 0
        self.background_elements = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.last_score_milestone = 0
        self.nectar_bank = 0
        self.game_over = False

        self.bee_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.bee_vel = np.array([0.0, 0.0], dtype=np.float32)

        self.card_hand = []
        self.active_effects = {}
        self.pollen_particles = []
        self.floating_texts = []
        
        self.last_space_held = False
        self.last_shift_held = False

        self._generate_flowers()
        self._generate_background()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        self.last_space_held, self.last_shift_held = space_held, shift_held

        # 1. Handle Player Input
        if space_pressed:
            reward += self._action_draw_card()
        if shift_pressed:
            reward += self._action_use_card()
        
        self._apply_movement(movement)

        # 2. Update Game Logic
        self._update_bee_physics()
        self._update_flowers()
        self._update_effects()
        self._update_particles()
        
        collision_reward = self._check_collisions()
        reward += collision_reward

        # 3. Update Game State
        self.steps += 1
        terminated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _apply_movement(self, movement):
        acceleration = self.BEE_ACCELERATION
        if "SPEED" in self.active_effects:
            acceleration *= 1.8 # Speed boost effect

        if movement == 1:  # Up
            self.bee_vel[1] -= acceleration
        elif movement == 2:  # Down
            self.bee_vel[1] += acceleration
        elif movement == 3:  # Left
            self.bee_vel[0] -= acceleration
        elif movement == 4:  # Right
            self.bee_vel[0] += acceleration

    def _update_bee_physics(self):
        # Apply friction
        self.bee_vel *= self.BEE_FRICTION
        
        # Clamp speed
        speed = np.linalg.norm(self.bee_vel)
        if speed > self.BEE_MAX_SPEED:
            self.bee_vel = self.bee_vel * (self.BEE_MAX_SPEED / speed)

        # Update position
        self.bee_pos += self.bee_vel

        # Boundary checks
        self.bee_pos[0] = np.clip(self.bee_pos[0], self.BEE_RADIUS, self.SCREEN_WIDTH - self.BEE_RADIUS)
        self.bee_pos[1] = np.clip(self.bee_pos[1], self.BEE_RADIUS, self.SCREEN_HEIGHT - self.BEE_RADIUS)

    def _update_flowers(self):
        for flower in self.flowers:
            if not flower["has_nectar"]:
                flower["respawn_timer"] -= 1
                if flower["respawn_timer"] <= 0:
                    flower["has_nectar"] = True

    def _update_effects(self):
        expired_effects = []
        for effect_type, data in self.active_effects.items():
            data["timer"] -= 1
            if data["timer"] <= 0:
                expired_effects.append(effect_type)
        for effect_type in expired_effects:
            del self.active_effects[effect_type]

    def _update_particles(self):
        self.pollen_particles = [p for p in self.pollen_particles if p["life"] > 0]
        for p in self.pollen_particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
        
        self.floating_texts = [t for t in self.floating_texts if t["life"] > 0]
        for t in self.floating_texts:
            t["pos"][1] -= 0.5
            t["life"] -= 1

    def _check_collisions(self):
        reward = 0.0
        collection_radius = self.BEE_RADIUS
        if "MAGNET" in self.active_effects:
            collection_radius *= 2.5 # Magnet effect

        for flower in self.flowers:
            if flower["has_nectar"]:
                dist = np.linalg.norm(self.bee_pos - flower["pos"])
                if dist < collection_radius + self.FLOWER_RADIUS:
                    # --- Nectar/Pollen Collection ---
                    flower["has_nectar"] = False
                    flower["respawn_timer"] = self.FLOWER_RESPAWN_TIME
                    
                    self.nectar_bank += 1
                    reward += 0.1 # Nectar reward
                    
                    pollen_gain = self.POLLEN_PER_FLOWER
                    self.score += pollen_gain
                    reward += 0.5 # Pollen reward
                    
                    # Milestone reward
                    if self.score // 100 > self.last_score_milestone:
                        self.last_score_milestone = self.score // 100
                        reward += 50
                        self._create_floating_text(f"+{50} BONUS!", self.bee_pos, color=(255,215,0))

                    # Visual feedback
                    self._create_pollen_burst(flower["pos"], 15)
                    self._create_floating_text(f"+{pollen_gain}", flower["pos"])
                    # sfx: collect_pollen.wav
        return reward

    def _action_draw_card(self):
        if self.nectar_bank > 0 and len(self.card_hand) < self.CARD_HAND_LIMIT:
            self.nectar_bank -= 1
            card_type = self.np_random.choice(list(self.CARDS.keys()))
            self.card_hand.append(card_type)
            # sfx: draw_card.wav
            return 1.0  # Reward for drawing a card
        return 0.0

    def _action_use_card(self):
        if self.card_hand:
            card_type = self.card_hand.pop(0)
            card_info = self.CARDS[card_type]
            self.active_effects[card_type] = {
                "timer": card_info["duration"],
                "color": card_info["color"]
            }
            self._create_pollen_burst(self.bee_pos, 30, color=card_info["color"])
            self._create_floating_text(f"{card_info['name']}!", self.bee_pos, color=card_info["color"])
            # sfx: use_card.wav
            return 0.0 # No direct reward for using a card, reward comes from its effect
        return 0.0

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "nectar": self.nectar_bank}

    def _generate_flowers(self):
        self.flowers = []
        min_dist = 2 * self.FLOWER_RADIUS + 20
        for _ in range(self.FLOWER_COUNT):
            while True:
                pos = self.np_random.uniform(
                    [self.FLOWER_RADIUS, self.FLOWER_RADIUS],
                    [self.SCREEN_WIDTH - self.FLOWER_RADIUS, self.SCREEN_HEIGHT - self.FLOWER_RADIUS - 50]
                )
                if not any(np.linalg.norm(pos - f["pos"]) < min_dist for f in self.flowers):
                    break
            self.flowers.append({
                "pos": pos,
                "has_nectar": True,
                "respawn_timer": 0,
            })

    def _generate_background(self):
        self.background_elements = []
        for _ in range(30):
            pos = self.np_random.uniform([0, 0], [self.SCREEN_WIDTH, self.SCREEN_HEIGHT])
            radius = self.np_random.uniform(10, 40)
            color = (
                self.np_random.integers(30, 70),
                self.np_random.integers(120, 180),
                self.np_random.integers(30, 70),
                self.np_random.integers(20, 50)
            )
            self.background_elements.append({"pos": pos, "radius": radius, "color": color})
            
    def _create_pollen_burst(self, pos, count, color=None):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.pollen_particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": self.np_random.integers(20, 40),
                "color": color if color else self.COLOR_POLLEN,
            })

    def _create_floating_text(self, text, pos, color=(255, 255, 255)):
        self.floating_texts.append({
            "pos": pos.copy(),
            "text": text,
            "life": 60,
            "color": color
        })

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        self.screen.fill(self.COLOR_GRASS, (0, self.SCREEN_HEIGHT - 50, self.SCREEN_WIDTH, 50))
        for element in self.background_elements:
            x, y = int(element["pos"][0]), int(element["pos"][1])
            r = int(element["radius"])
            pygame.gfxdraw.filled_circle(self.screen, x, y, r, element["color"])

    def _render_game(self):
        # Render Flowers
        for flower in self.flowers:
            x, y = int(flower["pos"][0]), int(flower["pos"][1])
            color = self.COLOR_FLOWER_FULL if flower["has_nectar"] else self.COLOR_FLOWER_EMPTY
            for i in range(5):
                angle = (i / 5) * 2 * math.pi + (self.steps / 20)
                px = x + int(math.cos(angle) * self.FLOWER_RADIUS)
                py = y + int(math.sin(angle) * self.FLOWER_RADIUS)
                pygame.gfxdraw.filled_circle(self.screen, px, py, self.FLOWER_RADIUS // 2, color)
                pygame.gfxdraw.aacircle(self.screen, px, py, self.FLOWER_RADIUS // 2, color)
            pygame.gfxdraw.filled_circle(self.screen, x, y, 7, self.COLOR_FLOWER_CENTER)

        # Render Particles
        for p in self.pollen_particles:
            alpha = int(255 * (p["life"] / 40))
            color = p["color"] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), 2, color)
            
        # Render Bee
        x, y = int(self.bee_pos[0]), int(self.bee_pos[1])
        
        # Active effect glows
        for effect_type, data in self.active_effects.items():
            radius_mult = 1.0 + math.sin(self.steps * 0.2) * 0.1
            glow_color = data["color"] + (80,)
            pygame.gfxdraw.filled_circle(self.screen, x, y, int(self.BEE_RADIUS * 2 * radius_mult), glow_color)

        # Base glow
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.BEE_RADIUS + 5, self.COLOR_BEE_GLOW)
        
        # Wings
        wing_angle = math.sin(self.steps * 0.8) * 0.8
        wing_len = self.BEE_RADIUS * 1.2
        pygame.draw.line(self.screen, self.COLOR_BEE_WING, (x, y), (x + int(math.cos(wing_angle) * wing_len), y + int(math.sin(wing_angle) * wing_len)), 5)
        pygame.draw.line(self.screen, self.COLOR_BEE_WING, (x, y), (x + int(math.cos(-wing_angle) * wing_len), y + int(math.sin(-wing_angle) * wing_len)), 5)

        # Body
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.BEE_RADIUS, self.COLOR_BEE)
        pygame.gfxdraw.aacircle(self.screen, x, y, self.BEE_RADIUS, (0,0,0))

        # Floating Texts
        for t in self.floating_texts:
            alpha = int(255 * (t["life"] / 60))
            text_surf = self.font_small.render(t["text"], True, t["color"])
            text_surf.set_alpha(alpha)
            self.screen.blit(text_surf, (int(t["pos"][0] - text_surf.get_width()/2), int(t["pos"][1])))

    def _render_ui(self):
        ui_surf = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        ui_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surf, (0, 0))

        # Pollen Score
        score_text = self.font_large.render(f"Pollen: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 8))

        # Timer
        remaining_time = (self.MAX_STEPS - self.steps) / 60.0
        timer_color = self.COLOR_TIMER_WARN if remaining_time < 10 else self.COLOR_TEXT
        timer_text = self.font_large.render(f"{remaining_time:.1f}", True, timer_color)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH/2 - timer_text.get_width()/2, 8))

        # Nectar & Cards
        nectar_text = self.font_small.render(f"Nectar: {self.nectar_bank}", True, self.COLOR_TEXT)
        self.screen.blit(nectar_text, (self.SCREEN_WIDTH - 220, 5))
        
        card_text = self.font_small.render("Cards:", True, self.COLOR_TEXT)
        self.screen.blit(card_text, (self.SCREEN_WIDTH - 220, 22))

        for i, card_type in enumerate(self.card_hand):
            card_info = self.CARDS[card_type]
            card_rect = pygame.Rect(self.SCREEN_WIDTH - 160 + i * 55, 8, 50, 24)
            pygame.draw.rect(self.screen, card_info["color"], card_rect, border_radius=4)
            card_name = self.font_small.render(card_info["name"], True, self.COLOR_TEXT)
            self.screen.blit(card_name, (card_rect.centerx - card_name.get_width()/2, card_rect.centery - card_name.get_height()/2))

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Play Controls ---
    # Arrows: Move
    # Space: Draw Card
    # Shift: Use Card
    
    # Setup Pygame for display
    pygame.display.set_caption("Bee Nectar Racer")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not done:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(60) # Limit to 60 FPS

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    pygame.quit()