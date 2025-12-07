import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An alchemist-themed 2D platformer.

    The agent controls an alchemist who must navigate a series of platforms.
    The goal is to collect colored ingredients (Red, Green, Blue), combine them
    at brewing stations into potions, and use the potion's special abilities
    (Jump Boost, Double Jump, Phasing) to reach the final alchemy lab.

    ### Action Space
    The action space is `MultiDiscrete([5, 2, 2])`.
    - `action[0]`: Movement (0: None, 1: Up/Jump, 2: Down, 3: Left, 4: Right)
    - `action[1]`: Brew Potion (0: Released, 1: Held) - Triggers on press at a station.
    - `action[2]`: Use Potion (0: Released, 1: Held) - Triggers on press.

    ### Observation Space
    The observation space is a `Box` with shape `(400, 640, 3)`, representing a
    640x400 RGB image of the game screen.

    ### Rewards
    - +100 for reaching the final lab (victory).
    - -100 for falling off the bottom of the screen (failure).
    - +5 for successfully phasing through a special platform.
    - +1 for brewing a potion.
    - +0.1 for collecting an ingredient.
    - -0.01 per step to encourage efficiency.

    ### Episode Termination
    - The episode ends if the alchemist reaches the lab.
    - The episode ends if the alchemist falls off the screen.
    - The episode ends if the maximum number of steps (2000) is reached.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a 2D world as an alchemist, collecting ingredients to brew powerful potions. "
        "Use your concoctions to jump higher, double jump, or phase through obstacles to reach the final lab."
    )
    user_guide = (
        "Controls: Use ←→ to move, ↑ to jump, and ↓ to drop through platforms. "
        "Press space at a brewing station to craft a potion, and shift to use it."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.W, self.H = 640, 400
        self.MAX_STEPS = 2000

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        
        # Game parameters
        self._setup_colors()
        self._setup_fonts()
        self._setup_game_parameters()
        
        # Initialize state variables - reset() is not called here because gym expects it to be called externally first
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.steps = 0

    def _setup_colors(self):
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_PLAYER = (230, 60, 160)
        self.COLOR_PLAYER_OUTLINE = (255, 200, 255)
        self.COLOR_PLATFORM = (100, 80, 120)
        self.COLOR_PLATFORM_PHASE = (100, 80, 120, 100) # RGBA for transparency
        self.COLOR_STATION = (200, 180, 50)
        self.COLOR_LAB = (50, 200, 180)
        self.COLOR_TEXT = (240, 240, 240)
        self.INGREDIENT_COLORS = {
            "red": (255, 70, 70), "green": (70, 255, 70), "blue": (70, 70, 255)
        }
        self.POTION_COLORS = {
            "JUMP_BOOST": (255, 255, 0), "DOUBLE_JUMP": (0, 255, 255), "PHASE": (255, 0, 255)
        }

    def _setup_fonts(self):
        self.font_main = pygame.font.Font(None, 24)
        self.font_hint = pygame.font.Font(None, 18)

    def _setup_game_parameters(self):
        self.GRAVITY = 0.4
        self.PLAYER_SPEED = 4.0
        self.JUMP_STRENGTH = -9.0
        self.JUMP_BOOST_STRENGTH = -12.0
        self.POTION_DURATION = 300 # 10 seconds at 30fps
        self.PLAYER_SIZE = (20, 30)
        
        self.RECIPES = {
            frozenset(["red", "green"]): "JUMP_BOOST",
            frozenset(["green", "blue"]): "DOUBLE_JUMP",
            frozenset(["red", "blue"]): "PHASE",
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Core game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Player state
        self.player_pos = pygame.Vector2(50, self.H - 100)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False
        self.is_dropping = False
        self.platform_stood_on = None

        # Inventory and Potions
        self.inventory = {"red": 0, "green": 0, "blue": 0}
        self.brewed_potion = None
        self.active_potion = None
        self.active_potion_timer = 0
        self.can_double_jump = False
        
        # Input state for edge detection
        self.prev_brew_held = False
        self.prev_use_held = False

        # Dynamic elements
        self.particles = []
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.platforms = [
            # Ground floor
            pygame.Rect(0, self.H - 40, 300, 40),
            # First jump
            pygame.Rect(350, self.H - 100, 150, 20),
            # Jump boost required
            pygame.Rect(150, self.H - 220, 150, 20),
            # Double jump required
            pygame.Rect(450, self.H - 300, 150, 20),
        ]
        self.phase_platforms = [
            pygame.Rect(0, self.H - 320, 150, 20)
        ]

        self.stations = [
            pygame.Rect(250, self.H - 60, 30, 20),
            pygame.Rect(170, self.H - 240, 30, 20)
        ]
        
        self.ingredients = [
            {"pos": pygame.Vector2(200, self.H - 60), "type": "red", "collected": False},
            {"pos": pygame.Vector2(400, self.H - 120), "type": "green", "collected": False},
            {"pos": pygame.Vector2(500, self.H - 120), "type": "blue", "collected": False},
            {"pos": pygame.Vector2(200, self.H - 240), "type": "red", "collected": False},
        ]
        
        self.lab_rect = pygame.Rect(30, self.H - 360, 60, 40)

    def step(self, action):
        reward = -0.01 # Small penalty for each step
        
        self._handle_input(action)
        reward += self._update_player_physics()
        interaction_reward, interaction_info = self._check_interactions()
        reward += interaction_reward
        self._update_game_state()
        
        self.score += reward
        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, brew_held, use_held = action[0], action[1] == 1, action[2] == 1
        
        # Horizontal movement
        if movement == 3: # Left
            self.player_vel.x = -self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_vel.x = self.PLAYER_SPEED
        else:
            self.player_vel.x = 0
            
        # Jump
        if movement == 1 and (self.on_ground or (self.active_potion == "DOUBLE_JUMP" and self.can_double_jump)):
            jump_strength = self.JUMP_BOOST_STRENGTH if self.active_potion == "JUMP_BOOST" else self.JUMP_STRENGTH
            self.player_vel.y = jump_strength
            if not self.on_ground:
                self.can_double_jump = False # Used the double jump
            self._add_particles(self.player_pos + pygame.Vector2(self.PLAYER_SIZE[0]/2, self.PLAYER_SIZE[1]), 10, (200,200,200))
            
        # Drop through platform
        if movement == 2 and self.on_ground and self.platform_stood_on:
            self.is_dropping = True
            self.player_pos.y += 2 # Nudge down to break collision

        # Brew Potion (on key press)
        if brew_held and not self.prev_brew_held:
            self.score += self._try_brew_potion()

        # Use Potion (on key press)
        if use_held and not self.prev_use_held:
            self._try_use_potion()
            
        self.prev_brew_held = brew_held
        self.prev_use_held = use_held
        
    def _update_player_physics(self):
        # Apply gravity
        self.player_vel.y += self.GRAVITY
        self.player_vel.y = min(self.player_vel.y, 10) # Terminal velocity

        # Move player
        self.player_pos += self.player_vel

        # Boundary checks
        self.player_pos.x = max(0, min(self.player_pos.x, self.W - self.PLAYER_SIZE[0]))
        if self.player_pos.y > self.H:
            self.game_over = True
            return -100 # Fell off screen

        # Collision detection
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, *self.PLAYER_SIZE)
        self.on_ground = False
        
        all_platforms = self.platforms
        if self.active_potion != "PHASE":
             all_platforms = self.platforms + self.phase_platforms

        for plat in all_platforms:
            if player_rect.colliderect(plat) and self.player_vel.y >= 0:
                if player_rect.bottom - self.player_vel.y <= plat.top + 1:
                    if self.is_dropping and self.platform_stood_on is plat:
                        continue
                    self.player_pos.y = plat.top - self.PLAYER_SIZE[1]
                    self.player_vel.y = 0
                    self.on_ground = True
                    self.is_dropping = False
                    self.platform_stood_on = plat
                    if self.active_potion == "DOUBLE_JUMP":
                        self.can_double_jump = True
                    break
        return 0

    def _check_interactions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, *self.PLAYER_SIZE)
        
        # Collect ingredients
        for ing in self.ingredients:
            if not ing["collected"]:
                ing_rect = pygame.Rect(ing["pos"].x - 5, ing["pos"].y - 5, 10, 10)
                if player_rect.colliderect(ing_rect):
                    ing["collected"] = True
                    self.inventory[ing["type"]] += 1
                    reward += 0.1
                    self._add_particles(ing["pos"], 15, self.INGREDIENT_COLORS[ing["type"]])

        # Phase through platforms
        if self.active_potion == "PHASE":
            for plat in self.phase_platforms:
                if player_rect.colliderect(plat):
                    reward += 5

        # Check victory condition
        if player_rect.colliderect(self.lab_rect):
            self.game_over = True
            reward += 100
        
        return reward, {}

    def _try_brew_potion(self):
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, *self.PLAYER_SIZE)
        at_station = any(player_rect.colliderect(s) for s in self.stations)

        if at_station and self.brewed_potion is None:
            collected_ingredients = frozenset(k for k, v in self.inventory.items() if v > 0)
            for recipe, potion_name in self.RECIPES.items():
                if recipe.issubset(collected_ingredients):
                    for item in recipe:
                        self.inventory[item] -= 1
                    
                    self.brewed_potion = potion_name
                    station_pos = next(s.center for s in self.stations if player_rect.colliderect(s))
                    self._add_particles(station_pos, 30, self.POTION_COLORS[potion_name], 2.0)
                    return 1.0
        return 0.0

    def _try_use_potion(self):
        if self.brewed_potion:
            self.active_potion = self.brewed_potion
            self.active_potion_timer = self.POTION_DURATION
            if self.active_potion == "DOUBLE_JUMP":
                self.can_double_jump = True
            self.brewed_potion = None
            player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, *self.PLAYER_SIZE)
            self._add_particles(player_rect.center, 40, self.POTION_COLORS[self.active_potion], 3.0, 0.5)

    def _update_game_state(self):
        # Update potion timer
        if self.active_potion_timer > 0:
            self.active_potion_timer -= 1
            if self.active_potion_timer == 0:
                self.active_potion = None
                self.can_double_jump = False
        
        # Update particles
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["radius"] *= 0.98

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_background_details()
        self._draw_platforms()
        self._draw_stations()
        self._draw_lab()
        self._draw_ingredients()
        self._update_and_draw_particles()
        self._draw_player()
        
    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"Score: {self.score:.2f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Steps
        steps_text = self.font_main.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 30))

        # Inventory
        inv_start_x = self.W - 120
        for i, (item, color) in enumerate(self.INGREDIENT_COLORS.items()):
            pygame.draw.rect(self.screen, color, (inv_start_x + i * 40, 10, 15, 15))
            count_text = self.font_main.render(f"{self.inventory[item]}", True, self.COLOR_TEXT)
            self.screen.blit(count_text, (inv_start_x + i * 40 + 20, 8))

        # Brewed Potion
        if self.brewed_potion:
            potion_text = self.font_main.render("Brewed:", True, self.COLOR_TEXT)
            self.screen.blit(potion_text, (self.W - 150, 40))
            pygame.draw.circle(self.screen, self.POTION_COLORS[self.brewed_potion], (self.W - 70, 50), 10)
            
        # Active Potion Timer
        if self.active_potion:
            timer_bar_width = self.PLAYER_SIZE[0] + 20
            filled_width = int(timer_bar_width * (self.active_potion_timer / self.POTION_DURATION))
            bar_pos_x = self.player_pos.x + self.PLAYER_SIZE[0]/2 - timer_bar_width/2
            pygame.draw.rect(self.screen, self.POTION_COLORS[self.active_potion], (bar_pos_x, self.player_pos.y - 15, filled_width, 5))
            pygame.draw.rect(self.screen, self.COLOR_TEXT, (bar_pos_x, self.player_pos.y - 15, timer_bar_width, 5), 1)

    def _draw_player(self):
        player_rect = pygame.Rect(int(self.player_pos.x), int(self.player_pos.y), *self.PLAYER_SIZE)
        
        # Active potion aura
        if self.active_potion:
            aura_color = self.POTION_COLORS[self.active_potion]
            aura_radius = int(self.PLAYER_SIZE[1] * 0.8 + math.sin(self.steps * 0.2) * 3)
            pygame.gfxdraw.filled_circle(self.screen, player_rect.centerx, player_rect.centery, aura_radius, (*aura_color, 50))
            pygame.gfxdraw.aacircle(self.screen, player_rect.centerx, player_rect.centery, aura_radius, (*aura_color, 100))

        # Player body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, player_rect, 1, border_radius=3)

    def _draw_platforms(self):
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat, border_radius=2)
        for plat in self.phase_platforms:
            s = pygame.Surface(plat.size, pygame.SRCALPHA)
            color = self.POTION_COLORS["PHASE"] if self.active_potion == "PHASE" else self.COLOR_PLATFORM
            alpha = 200 if self.active_potion == "PHASE" else 100
            pygame.draw.rect(s, (*color, alpha), s.get_rect(), border_radius=2)
            self.screen.blit(s, plat.topleft)

    def _draw_stations(self):
        for station in self.stations:
            pygame.draw.rect(self.screen, self.COLOR_STATION, station, border_radius=3)
            pygame.draw.rect(self.screen, (255,255,255), station, 1, border_radius=3)
            hint_text = self.font_hint.render("Brew", True, self.COLOR_TEXT)
            self.screen.blit(hint_text, (station.centerx - 15, station.y - 15))

    def _draw_ingredients(self):
        for ing in self.ingredients:
            if not ing["collected"]:
                pos = ing["pos"]
                color = self.INGREDIENT_COLORS[ing["type"]]
                y_offset = math.sin(self.steps * 0.1 + pos.x) * 3
                pygame.draw.circle(self.screen, color, (int(pos.x), int(pos.y + y_offset)), 6)
                pygame.draw.circle(self.screen, (255,255,255), (int(pos.x), int(pos.y + y_offset)), 6, 1)

    def _draw_lab(self):
        pygame.draw.rect(self.screen, self.COLOR_LAB, self.lab_rect, border_radius=5)
        pygame.draw.rect(self.screen, (255,255,255), self.lab_rect, 2, border_radius=5)
        lab_text = self.font_hint.render("GOAL", True, self.COLOR_BG)
        self.screen.blit(lab_text, (self.lab_rect.centerx - 15, self.lab_rect.centery - 8))
        
    def _draw_background_details(self):
        for i in range(10):
            x = (self.steps * 0.1 + i * 100) % (self.W + 200) - 100
            y = (i * 137) % self.H
            size = (i % 3 + 1) * 5
            color = (
                self.COLOR_BG[0] + (i % 4) * 5,
                self.COLOR_BG[1] + (i % 4) * 5,
                self.COLOR_BG[2] + (i % 4) * 10,
            )
            pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), int(size), color)

    def _add_particles(self, pos, count, color, speed_mult=1.0, life_mult=1.0):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) * speed_mult
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": random.randint(20, 40) * life_mult,
                "color": color,
                "radius": random.uniform(2, 5)
            })

    def _update_and_draw_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["life"] / 60))
            alpha = max(0, min(255, alpha))
            color_with_alpha = (*p["color"], alpha)
            try:
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), color_with_alpha)
            except (OverflowError, ValueError):
                pass

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "inventory": self.inventory,
            "active_potion": self.active_potion or "None",
        }

    def close(self):
        pygame.quit()
        

if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not be run when the environment is used by an agent
    
    # Un-comment the line below to run with a display
    os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.W, env.H))
    pygame.display.set_caption("Alchemist Platformer")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        brew = 0
        use = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        if keys[pygame.K_RIGHT]:
            movement = 4
        if keys[pygame.K_UP]:
            movement = 1
        if keys[pygame.K_DOWN]:
            movement = 2
        if keys[pygame.K_SPACE]:
            brew = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            use = 1
            
        action = [movement, brew, use]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished! Total reward: {total_reward}")
            print(f"Info: {info}")
            total_reward = 0
            obs, info = env.reset()
            pygame.time.wait(2000)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(30) # Run at 30 FPS
        
    env.close()