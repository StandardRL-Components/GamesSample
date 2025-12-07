import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:00:12.132837
# Source Brief: brief_01456.md
# Brief Index: 1456
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment where the player controls a platform to launch projectiles
    at descending enemies. The core mechanic involves creating chain reactions by
    colliding projectiles to increase their speed and power.

    **Visuals:**
    - Dark background with a subtle starfield for depth.
    - Player platform is a bright white rectangle.
    - Enemies are vibrant red squares.
    - Projectiles are glowing circles that change color from yellow to red as they speed up.
    - Explosions, particle trails, and text popups provide satisfying feedback.

    **Gameplay:**
    - **Goal:** Destroy 40 enemies within 90 seconds.
    - **Controls:** Move the platform left/right. Press space to launch a projectile.
    - **Chain Reactions:** When a projectile hits another, both get a speed boost.
    - **Scoring:** Points are awarded for survival, hitting enemies, and creating chains.
      A large bonus/penalty is given for winning/losing.
    - **Difficulty:** The rate of enemy spawns increases over time.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a platform to launch projectiles at descending enemies. "
        "Create chain reactions by colliding projectiles to increase their power and score."
    )
    user_guide = "Controls: ←→ to move the platform. Press space to launch a projectile."
    auto_advance = True

    # --- CONSTANTS ---
    # Game parameters
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    GAME_DURATION_SECONDS = 90
    MAX_STEPS = GAME_DURATION_SECONDS * FPS
    WIN_CONDITION_ENEMIES = 40

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_WALL = (100, 100, 120)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_PROJECTILE_SLOW = pygame.Color(255, 255, 0)
    COLOR_PROJECTILE_FAST = pygame.Color(255, 100, 0)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_POPUP = (255, 255, 100)

    # Player Platform
    PLAYER_WIDTH = 80
    PLAYER_HEIGHT = 10
    PLAYER_Y = SCREEN_HEIGHT - 30
    PLAYER_SPEED = 8

    # Projectiles
    PROJECTILE_RADIUS = 6
    PROJECTILE_BASE_SPEED = 4
    PROJECTILE_MAX_SPEED = 12
    PROJECTILE_SPEED_BOOST = 1.2
    PROJECTILE_LAUNCH_COOLDOWN = 15 # frames

    # Enemies
    ENEMY_SIZE = 20
    ENEMY_BASE_SPEED = 1.0
    ENEMY_SPAWN_INTERVAL_INITIAL = 2.0 * FPS
    ENEMY_SPAWN_RATE_INCREASE_INTERVAL = 5.0 * FPS
    ENEMY_SPAWN_RATE_MULTIPLIER = 0.99 # a 1% increase in rate is a 1% decrease in interval

    # Rewards
    REWARD_SURVIVAL = 0.001 # Per step
    REWARD_ENEMY_ELIMINATED = 5.0
    REWARD_CHAIN_REACTION = 1.0
    REWARD_WIN = 100.0
    REWARD_LOSS = -100.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_popup = pygame.font.SysFont("monospace", 22, bold=True)
        self.font_combo = pygame.font.SysFont("impact", 40)

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.platform_pos = None
        self.projectiles = []
        self.enemies = []
        self.explosions = []
        self.particles = []
        self.text_popups = []
        self.enemies_eliminated = 0
        self.enemy_spawn_timer = 0
        self.enemy_spawn_interval = 0
        self.spawn_rate_update_timer = 0
        self.launch_cooldown_timer = 0
        self.space_was_held = False
        self.combo_count = 0
        self.combo_display_timer = 0

        # Background stars for parallax effect
        self.stars = [
            {
                "pos": [random.uniform(0, self.SCREEN_WIDTH), random.uniform(0, self.SCREEN_HEIGHT)],
                "radius": random.uniform(0.5, 1.5),
                "speed": random.uniform(0.1, 0.3)
            } for _ in range(100)
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.platform_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.PLAYER_Y)
        self.projectiles.clear()
        self.enemies.clear()
        self.explosions.clear()
        self.particles.clear()
        self.text_popups.clear()
        self.enemies_eliminated = 0
        
        self.enemy_spawn_interval = self.ENEMY_SPAWN_INTERVAL_INITIAL
        self.enemy_spawn_timer = self.enemy_spawn_interval
        self.spawn_rate_update_timer = 0
        
        self.launch_cooldown_timer = 0
        self.space_was_held = False
        self.combo_count = 0
        self.combo_display_timer = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = self.REWARD_SURVIVAL

        # --- Game Logic ---
        self._handle_input(action)
        self._update_game_state()
        reward += self._handle_collisions()
        self._spawn_enemies()

        # --- Termination Check ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.enemies_eliminated >= self.WIN_CONDITION_ENEMIES:
                reward += self.REWARD_WIN # Win reward
            else:
                reward += self.REWARD_LOSS # Loss reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # Platform movement
        if movement == 3: # Left
            self.platform_pos.x -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.platform_pos.x += self.PLAYER_SPEED
        self.platform_pos.x = np.clip(self.platform_pos.x, self.PLAYER_WIDTH / 2, self.SCREEN_WIDTH - self.PLAYER_WIDTH / 2)

        # Projectile launch (on key press, not hold)
        if self.launch_cooldown_timer > 0:
            self.launch_cooldown_timer -= 1
            
        if space_held and not self.space_was_held and self.launch_cooldown_timer == 0:
            self._launch_projectile()
            self.launch_cooldown_timer = self.PROJECTILE_LAUNCH_COOLDOWN
            # sfx: player_shoot.wav
        
        self.space_was_held = space_held

    def _launch_projectile(self):
        proj = {
            "pos": pygame.Vector2(self.platform_pos.x, self.platform_pos.y - self.PLAYER_HEIGHT),
            "vel": pygame.Vector2(0, -self.PROJECTILE_BASE_SPEED),
            "speed": self.PROJECTILE_BASE_SPEED,
            "radius": self.PROJECTILE_RADIUS,
            "hits": 0,
        }
        self.projectiles.append(proj)
        self.combo_count = 0

    def _update_game_state(self):
        # Update timers
        self.enemy_spawn_timer += 1
        self.spawn_rate_update_timer += 1
        if self.combo_display_timer > 0:
            self.combo_display_timer -= 1

        # Increase difficulty over time
        if self.spawn_rate_update_timer >= self.ENEMY_SPAWN_RATE_INCREASE_INTERVAL:
            self.spawn_rate_update_timer = 0
            self.enemy_spawn_interval *= self.ENEMY_SPAWN_RATE_MULTIPLIER
            self.enemy_spawn_interval = max(self.enemy_spawn_interval, 15) # Set a minimum spawn interval

        # Update projectiles
        for p in self.projectiles[:]:
            p["pos"] += p["vel"]
            # Wall bounces
            if p["pos"].x < p["radius"] or p["pos"].x > self.SCREEN_WIDTH - p["radius"]:
                p["vel"].x *= -1
                p["pos"].x = np.clip(p["pos"].x, p["radius"], self.SCREEN_WIDTH - p["radius"])
                # sfx: bounce.wav
            # Remove if off-screen (top) or max speed reached
            if p["pos"].y < -p["radius"] or p["speed"] >= self.PROJECTILE_MAX_SPEED:
                self.projectiles.remove(p)

        # Update enemies
        for e in self.enemies[:]:
            e["pos"].y += self.ENEMY_BASE_SPEED
            if e["pos"].y > self.SCREEN_HEIGHT + e["size"]:
                self.enemies.remove(e)

        # Update effects
        for exp in self.explosions[:]:
            exp["lifetime"] -= 1
            exp["radius"] += 2
            if exp["lifetime"] <= 0:
                self.explosions.remove(exp)
        
        for part in self.particles[:]:
            part["pos"] += part["vel"]
            part["lifetime"] -= 1
            if part["lifetime"] <= 0:
                self.particles.remove(part)

        for pop in self.text_popups[:]:
            pop["pos"].y -= 0.5
            pop["lifetime"] -= 1
            if pop["lifetime"] <= 0:
                self.text_popups.remove(pop)

    def _spawn_enemies(self):
        if self.enemy_spawn_timer >= self.enemy_spawn_interval:
            self.enemy_spawn_timer = 0
            spawn_x = random.uniform(self.ENEMY_SIZE, self.SCREEN_WIDTH - self.ENEMY_SIZE)
            enemy = {
                "pos": pygame.Vector2(spawn_x, -self.ENEMY_SIZE),
                "size": self.ENEMY_SIZE,
                "rect": pygame.Rect(0, 0, self.ENEMY_SIZE, self.ENEMY_SIZE)
            }
            self.enemies.append(enemy)

    def _handle_collisions(self):
        reward = 0
        
        # Projectile-Enemy collisions
        for p in self.projectiles[:]:
            proj_rect = pygame.Rect(p["pos"].x - p["radius"], p["pos"].y - p["radius"], p["radius"]*2, p["radius"]*2)
            for e in self.enemies[:]:
                e["rect"].center = e["pos"]
                if proj_rect.colliderect(e["rect"]):
                    self._create_explosion(e["pos"])
                    self.enemies.remove(e)
                    if p in self.projectiles: self.projectiles.remove(p)
                    
                    self.enemies_eliminated += 1
                    self.score += 100 * (1 + self.combo_count)
                    reward += self.REWARD_ENEMY_ELIMINATED
                    
                    self.combo_count += 1
                    self.combo_display_timer = 60
                    
                    # sfx: explosion.wav
                    break # Projectile is used up

        # Projectile-Projectile collisions
        collided_pairs = set()
        for i in range(len(self.projectiles)):
            for j in range(i + 1, len(self.projectiles)):
                p1 = self.projectiles[i]
                p2 = self.projectiles[j]
                dist = p1["pos"].distance_to(p2["pos"])
                if dist < p1["radius"] + p2["radius"]:
                    if (i, j) not in collided_pairs:
                        self._boost_projectile(p1)
                        self._boost_projectile(p2)
                        
                        # Create visual feedback for chain reaction
                        mid_point = p1["pos"].lerp(p2["pos"], 0.5)
                        self._create_text_popup("CHAIN!", mid_point)
                        self._create_chain_particles(mid_point)
                        
                        reward += self.REWARD_CHAIN_REACTION
                        self.score += 50
                        collided_pairs.add((i, j))
                        # sfx: chain_reaction.wav
        return reward
    
    def _boost_projectile(self, p):
        p["speed"] = min(self.PROJECTILE_MAX_SPEED, p["speed"] * self.PROJECTILE_SPEED_BOOST)
        p["vel"].scale_to_length(p["speed"])

    def _create_explosion(self, pos):
        self.explosions.append({"pos": pos, "radius": 5, "lifetime": 15})
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pos.copy(), "vel": vel, "lifetime": random.randint(20, 40),
                "color": random.choice([(255,255,255), (255,200,0)])
            })

    def _create_chain_particles(self, pos):
        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pos.copy(), "vel": vel, "lifetime": random.randint(15, 30),
                "color": self.COLOR_PROJECTILE_SLOW
            })

    def _create_text_popup(self, text, pos):
        self.text_popups.append({
            "text": text, "pos": pos.copy(), "lifetime": 45, "color": self.COLOR_TEXT_POPUP
        })

    def _check_termination(self):
        return (self.steps >= self.MAX_STEPS or
                self.enemies_eliminated >= self.WIN_CONDITION_ENEMIES)

    def _get_observation(self):
        self._render_frame()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "enemies_eliminated": self.enemies_eliminated,
            "time_remaining": (self.MAX_STEPS - self.steps) / self.FPS
        }

    def _render_frame(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_walls()
        self._render_entities()
        self._render_effects()
        self._render_ui()

    def _render_background(self):
        for star in self.stars:
            star["pos"][1] += star["speed"]
            if star["pos"][1] > self.SCREEN_HEIGHT:
                star["pos"][0] = random.uniform(0, self.SCREEN_WIDTH)
                star["pos"][1] = 0
            pygame.draw.circle(self.screen, (200, 200, 220), star["pos"], star["radius"])

    def _render_walls(self):
        pygame.draw.line(self.screen, self.COLOR_WALL, (0,0), (0, self.SCREEN_HEIGHT), 2)
        pygame.draw.line(self.screen, self.COLOR_WALL, (self.SCREEN_WIDTH-1,0), (self.SCREEN_WIDTH-1, self.SCREEN_HEIGHT), 2)

    def _render_entities(self):
        # Player
        player_rect = pygame.Rect(0, 0, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        player_rect.center = self.platform_pos
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
        # Enemies
        for e in self.enemies:
            e["rect"].center = e["pos"]
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, e["rect"])
            
        # Projectiles
        for p in self.projectiles:
            progress = (p["speed"] - self.PROJECTILE_BASE_SPEED) / (self.PROJECTILE_MAX_SPEED - self.PROJECTILE_BASE_SPEED)
            progress = max(0, min(1, progress))
            color = self.COLOR_PROJECTILE_SLOW.lerp(self.COLOR_PROJECTILE_FAST, progress)
            self._draw_glowing_circle(self.screen, color, p["pos"], p["radius"])

    def _render_effects(self):
        # Particles
        for part in self.particles:
            size = max(0, part["lifetime"] / 20.0)
            pygame.draw.circle(self.screen, part["color"], part["pos"], size)

        # Explosions
        for exp in self.explosions:
            alpha = max(0, 255 * (exp["lifetime"] / 15.0))
            color = (255, 255, 255, alpha)
            self._draw_glowing_circle(self.screen, color, exp["pos"], exp["radius"], is_explosion=True)

        # Text Popups
        for pop in self.text_popups:
            alpha = max(0, min(255, pop["lifetime"] * 6))
            text_surf = self.font_popup.render(pop["text"], True, pop["color"])
            text_surf.set_alpha(alpha)
            text_rect = text_surf.get_rect(center=pop["pos"])
            self.screen.blit(text_surf, text_rect)
            
        # Combo counter
        if self.combo_display_timer > 0 and self.combo_count > 1:
            alpha = min(255, self.combo_display_timer * 5)
            combo_text = f"x{self.combo_count}"
            text_surf = self.font_combo.render(combo_text, True, self.COLOR_TEXT_POPUP)
            text_surf.set_alpha(alpha)
            text_rect = text_surf.get_rect(midbottom=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT - 50))
            self.screen.blit(text_surf, text_rect)


    def _render_ui(self):
        # Time remaining
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = f"TIME: {time_left:.1f}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (10, 10))
        
        # Enemies eliminated
        enemies_text = f"KILLS: {self.enemies_eliminated}/{self.WIN_CONDITION_ENEMIES}"
        enemies_surf = self.font_ui.render(enemies_text, True, self.COLOR_TEXT)
        enemies_rect = enemies_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(enemies_surf, enemies_rect)
        
        # Score
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(midbottom=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 5))
        self.screen.blit(score_surf, score_rect)

    def _draw_glowing_circle(self, surface, color, center, radius, is_explosion=False):
        center_int = (int(center.x), int(center.y))
        
        # Create a temporary surface for the glow effect
        glow_radius = int(radius * 2.5)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        
        # Draw concentric circles with decreasing alpha for the glow
        for i in range(glow_radius, 0, -2):
            alpha = 30 * (1 - i / glow_radius)
            if is_explosion:
                alpha = 80 * (1 - i/glow_radius)
            
            pygame.gfxdraw.aacircle(
                glow_surf, glow_radius, glow_radius, i, (*color[:3], int(alpha))
            )
        
        # Blit the glow surface onto the main screen
        surface.blit(glow_surf, (center_int[0] - glow_radius, center_int[1] - glow_radius))

        # Draw the solid inner circle
        pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], int(radius), color)
        pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], int(radius), color)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # --- Manual Play Example ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Override Pygame screen for direct display
    os.environ["SDL_VIDEODRIVER"] = "x11"
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Chain Reaction Command")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    print("\n--- Controls ---")
    print("Left/Right Arrow: Move Platform")
    print("Spacebar: Launch Projectile")
    print("Q or ESC: Quit")
    print("R: Reset Environment")
    
    while not terminated:
        # Action defaults
        movement = 0 # none
        space_pressed = 0 # released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    terminated = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Environment Reset ---")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        if keys[pygame.K_SPACE]:
            space_pressed = 1
            
        action = [movement, space_pressed, 0] # shift is unused
        
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        
        if term:
            print(f"--- Episode Finished ---")
            print(f"Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            print(f"Kills: {info['enemies_eliminated']}/{GameEnv.WIN_CONDITION_ENEMIES}")
            if info['enemies_eliminated'] >= GameEnv.WIN_CONDITION_ENEMIES:
                print("Result: VICTORY!")
            else:
                print("Result: Defeat (Time Up)")
            
            # Wait for reset
            waiting_for_reset = True
            while waiting_for_reset:
                 for event in pygame.event.get():
                     if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and (event.key == pygame.K_q or event.key == pygame.K_ESCAPE)):
                         terminated = True
                         waiting_for_reset = False
                     if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                         obs, info = env.reset()
                         total_reward = 0
                         print("\n--- Environment Reset ---")
                         waiting_for_reset = False
                 # Keep rendering final screen
                 surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                 screen.blit(surf, (0, 0))
                 pygame.display.flip()
                 clock.tick(env.FPS)


        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    env.close()