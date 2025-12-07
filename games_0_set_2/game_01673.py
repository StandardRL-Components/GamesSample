
# Generated: 2025-08-28T02:19:39.550362
# Source Brief: brief_01673.md
# Brief Index: 1673

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to move. Hold Space to fire. Hold Shift for a temporary shield (has a cooldown)."
    )

    game_description = (
        "A fast-paced, retro arcade space shooter. Destroy all 50 alien invaders before they deplete your shields. Chain kills for bonus points!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1500 # Extended to allow for completion
        self.NUM_ALIENS = 50
        self.NUM_STARS = 100

        # Colors
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_PROJECTILE = (100, 200, 255)
        self.COLOR_ENEMY_PROJECTILE = (255, 100, 100)
        self.COLOR_SHIELD_BAR = (0, 255, 255)
        self.COLOR_SHIELD_ACTIVE = (200, 255, 255, 100)
        self.COLOR_TEXT = (220, 220, 220)
        self.ALIEN_COLORS = {
            1: (200, 100, 255), # Purple
            2: (255, 150, 50),  # Orange
            3: (220, 220, 220)  # White
        }

        # Reward structure
        self.REWARD_PER_STEP = -0.01
        self.REWARD_KILL_ALIEN = 1.0
        self.REWARD_KILL_CHAIN = 5.0
        self.REWARD_LOSE_SHIELD = -10.0
        self.REWARD_WIN = 100.0
        self.REWARD_LOSE = -100.0

        # Gymnasium spaces
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        
        # Initialize state variables (will be populated in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player = None
        self.aliens = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.stars = []
        self.kill_chain_count = 0
        self.kill_chain_timer = 0
        self.popup_texts = []
        self.alien_fire_chance = 0.0
        self.np_random = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Player state
        self.player = {
            "rect": pygame.Rect(self.WIDTH / 2 - 15, self.HEIGHT - 50, 30, 20),
            "shields": 3,
            "fire_cooldown": 0,
            "shield_cooldown": 0,
            "shield_active_timer": 0,
            "speed": 7,
            "hit_timer": 0
        }
        
        # Game entities
        self.aliens = self._spawn_aliens()
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.popup_texts = []
        
        # Background
        self.stars = [
            (
                self.np_random.integers(0, self.WIDTH),
                self.np_random.integers(0, self.HEIGHT),
                self.np_random.integers(1, 4),
                self.np_random.integers(50, 150),
            )
            for _ in range(self.NUM_STARS)
        ]

        # Difficulty and scoring
        self.kill_chain_count = 0
        self.kill_chain_timer = 0
        self.alien_fire_chance = 0.002 # Initial chance per alien per frame
        
        return self._get_observation(), self._get_info()

    def _spawn_aliens(self):
        aliens = []
        rows = 5
        cols = self.NUM_ALIENS // rows
        for i in range(self.NUM_ALIENS):
            row = i // cols
            col = i % cols
            
            alien_type = (row % 3) + 1
            
            x = col * 45 + (self.WIDTH - cols * 45) / 2
            y = row * 40 + 40
            
            alien = {
                "rect": pygame.Rect(x, y, 25, 25),
                "type": alien_type,
                "direction": 1,
                "initial_x": x
            }
            aliens.append(alien)
        return aliens

    def step(self, action):
        reward = self.REWARD_PER_STEP
        self.game_over = self._check_termination()
        
        if not self.game_over:
            # Unpack action
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            
            # Update timers
            self._update_timers()
            
            # Handle player input
            self._handle_input(movement, space_held, shift_held)
            
            # Update game state
            self._update_projectiles()
            self._update_aliens()
            
            # Handle collisions and events
            reward += self._handle_collisions()
            
            # Update particles and popups
            self._update_effects()
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated and not self.game_over:
            if not self.aliens:
                reward += self.REWARD_WIN
                self.popup_texts.append(self._create_popup("VICTORY!", self.WIDTH/2, self.HEIGHT/2, 60, (100, 255, 100)))
            elif self.player["shields"] <= 0:
                reward += self.REWARD_LOSE
                self.popup_texts.append(self._create_popup("DEFEAT", self.WIDTH/2, self.HEIGHT/2, 60, (255, 100, 100)))
            self.game_over = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_timers(self):
        if self.player["fire_cooldown"] > 0: self.player["fire_cooldown"] -= 1
        if self.player["shield_cooldown"] > 0: self.player["shield_cooldown"] -= 1
        if self.player["shield_active_timer"] > 0: self.player["shield_active_timer"] -= 1
        if self.player["hit_timer"] > 0: self.player["hit_timer"] -= 1
        if self.kill_chain_timer > 0:
            self.kill_chain_timer -= 1
        elif self.kill_chain_count > 0:
            self.kill_chain_count = 0

    def _handle_input(self, movement, space_held, shift_held):
        # Movement
        if movement == 1: self.player["rect"].y -= self.player["speed"]
        if movement == 2: self.player["rect"].y += self.player["speed"]
        if movement == 3: self.player["rect"].x -= self.player["speed"]
        if movement == 4: self.player["rect"].x += self.player["speed"]
        self.player["rect"].clamp_ip(self.screen.get_rect())

        # Firing
        if space_held and self.player["fire_cooldown"] == 0:
            # sfx: player_shoot.wav
            proj_rect = pygame.Rect(self.player["rect"].centerx - 2, self.player["rect"].top - 10, 4, 12)
            self.player_projectiles.append({"rect": proj_rect, "speed": -12})
            self.player["fire_cooldown"] = 5 # 6 shots per second

        # Shield
        if shift_held and self.player["shield_cooldown"] == 0:
            # sfx: shield_activate.wav
            self.player["shield_active_timer"] = 30 # 1 second duration
            self.player["shield_cooldown"] = 150 # 5 second cooldown

    def _update_projectiles(self):
        for p in self.player_projectiles: p["rect"].y += p["speed"]
        for p in self.enemy_projectiles: p["rect"].y += p["speed"]
        self.player_projectiles = [p for p in self.player_projectiles if p["rect"].bottom > 0]
        self.enemy_projectiles = [p for p in self.enemy_projectiles if p["rect"].top < self.HEIGHT]

    def _update_aliens(self):
        # Increase difficulty
        if self.steps > 0 and self.steps % 100 == 0:
            self.alien_fire_chance += 0.0005

        for a in self.aliens:
            # Movement
            if a["type"] == 1: # Horizontal
                a["rect"].x += a["direction"] * 2
                if abs(a["rect"].x - a["initial_x"]) > 50: a["direction"] *= -1
            elif a["type"] == 2: # Vertical (sin wave)
                a["rect"].y = 100 + math.sin(self.steps / 30 + a["initial_x"] / 50) * 40
            elif a["type"] == 3: # Diagonal
                a["rect"].x += a["direction"]
                a["rect"].y += 1
                if a["rect"].left < 0 or a["rect"].right > self.WIDTH: a["direction"] *= -1
                if a["rect"].top > self.HEIGHT: a["rect"].y = 0

            # Firing
            if self.np_random.random() < self.alien_fire_chance:
                # sfx: enemy_shoot.wav
                proj_rect = pygame.Rect(a["rect"].centerx - 3, a["rect"].bottom, 6, 6)
                self.enemy_projectiles.append({"rect": proj_rect, "speed": 5})

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs Aliens
        for p in self.player_projectiles[:]:
            hit_alien = p["rect"].collidelist([a["rect"] for a in self.aliens])
            if hit_alien != -1:
                # sfx: explosion.wav
                self.player_projectiles.remove(p)
                destroyed_alien = self.aliens.pop(hit_alien)
                self._create_explosion(destroyed_alien["rect"].center, self.ALIEN_COLORS[destroyed_alien["type"]])
                
                self.score += 10
                reward += self.REWARD_KILL_ALIEN
                
                # Kill chain
                self.kill_chain_timer = 90 # 3 seconds to get next kill
                self.kill_chain_count += 1
                if self.kill_chain_count >= 3:
                    reward += self.REWARD_KILL_CHAIN
                    self.score += 20 * self.kill_chain_count
                    popup_text = f"CHAIN x{self.kill_chain_count}!"
                    self.popup_texts.append(self._create_popup(popup_text, destroyed_alien["rect"].centerx, destroyed_alien["rect"].centery))
                break

        # Enemy projectiles vs Player
        if self.player["shield_active_timer"] == 0 and self.player["hit_timer"] == 0:
            for p in self.enemy_projectiles[:]:
                if self.player["rect"].colliderect(p["rect"]):
                    # sfx: player_hit.wav
                    self.enemy_projectiles.remove(p)
                    self.player["shields"] -= 1
                    self.player["hit_timer"] = 60 # 2s invulnerability
                    reward += self.REWARD_LOSE_SHIELD
                    self._create_explosion(self.player["rect"].center, self.COLOR_PLAYER, 40)
                    break
        
        return reward

    def _update_effects(self):
        # Particles
        for part in self.particles[:]:
            part["pos"][0] += part["vel"][0]
            part["pos"][1] += part["vel"][1]
            part["life"] -= 1
            part["vel"][1] += 0.05 # Gravity
            if part["life"] <= 0: self.particles.remove(part)
        
        # Popups
        for pop in self.popup_texts[:]:
            pop["pos"][1] -= 0.5
            pop["life"] -= 1
            if pop["life"] <= 0: self.popup_texts.remove(pop)

    def _create_explosion(self, pos, color, num_particles=20):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(20, 40),
                "color": color,
                "radius": self.np_random.uniform(2, 5)
            })
    
    def _create_popup(self, text, x, y, life=45, color=None):
        if color is None: color = self.COLOR_TEXT
        return {"text": text, "pos": [x, y], "life": life, "color": color}

    def _check_termination(self):
        return (
            self.player["shields"] <= 0 or
            not self.aliens or
            self.steps >= self.MAX_STEPS
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "shields": self.player["shields"]}

    def _render_game(self):
        # Stars
        for x, y, size, brightness in self.stars:
            pygame.draw.circle(self.screen, (brightness, brightness, brightness), (x, y), size / 2)

        # Aliens
        for a in self.aliens:
            color = self.ALIEN_COLORS[a["type"]]
            if a["type"] == 1: pygame.draw.rect(self.screen, color, a["rect"])
            elif a["type"] == 2: pygame.gfxdraw.filled_circle(self.screen, a["rect"].centerx, a["rect"].centery, a["rect"].width // 2, color)
            elif a["type"] == 3:
                pts = [a["rect"].midtop, a["rect"].midright, a["rect"].midbottom, a["rect"].midleft]
                pygame.gfxdraw.filled_polygon(self.screen, pts, color)

        # Player
        if self.player["shields"] > 0:
            color = self.COLOR_PLAYER if self.player["hit_timer"] // 5 % 2 == 0 else (255, 255, 255)
            pts = [self.player["rect"].midtop, self.player["rect"].bottomleft, self.player["rect"].bottomright]
            pygame.gfxdraw.aapolygon(self.screen, pts, color)
            pygame.gfxdraw.filled_polygon(self.screen, pts, color)
            # Active Shield visual
            if self.player["shield_active_timer"] > 0:
                alpha = 50 + (self.player["shield_active_timer"] / 30) * 100
                shield_surface = pygame.Surface((60, 60), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(shield_surface, 30, 30, 30, (200, 255, 255, int(alpha)))
                pygame.gfxdraw.aacircle(shield_surface, 30, 30, 30, (255, 255, 255, int(alpha)))
                self.screen.blit(shield_surface, (self.player["rect"].centerx - 30, self.player["rect"].centery - 30))

        # Projectiles
        for p in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJECTILE, p["rect"])
            pygame.draw.line(self.screen, (255,255,255), p["rect"].midbottom, p["rect"].midtop, 2)
        for p in self.enemy_projectiles:
            pygame.gfxdraw.filled_circle(self.screen, p["rect"].centerx, p["rect"].centery, p["rect"].width // 2, self.COLOR_ENEMY_PROJECTILE)

        # Particles
        for part in self.particles:
            size = part["radius"] * (part["life"] / 40.0)
            if size > 0:
                pygame.draw.circle(self.screen, part["color"], part["pos"], int(size))

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Shields
        for i in range(self.player["shields"]):
            shield_rect = pygame.Rect(self.WIDTH - 30 - i * 35, 15, 25, 15)
            pts = [shield_rect.midtop, shield_rect.bottomleft, shield_rect.bottomright]
            pygame.gfxdraw.filled_polygon(self.screen, pts, self.COLOR_SHIELD_BAR)
            pygame.gfxdraw.aapolygon(self.screen, pts, (255,255,255))
        
        # Shield Cooldown
        if self.player["shield_cooldown"] > 0:
            cooldown_ratio = self.player["shield_cooldown"] / 150
            bar_width = 100
            bar_height = 10
            bar_x = self.WIDTH - bar_width - 10
            bar_y = 45
            pygame.draw.rect(self.screen, (50,50,80), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_SHIELD_BAR, (bar_x, bar_y, bar_width * (1-cooldown_ratio), bar_height))

        # Popups
        for pop in self.popup_texts:
            alpha = min(255, pop["life"] * 10)
            pop_font = self.font_small if "CHAIN" in pop["text"] else self.font_main
            pop_surf = pop_font.render(pop["text"], True, pop["color"])
            pop_surf.set_alpha(alpha)
            pop_rect = pop_surf.get_rect(center=(pop["pos"][0], pop["pos"][1]))
            self.screen.blit(pop_surf, pop_rect)

        # Game Over message
        if self.game_over and self.steps > self.MAX_STEPS:
            self.popup_texts.append(self._create_popup("TIME UP", self.WIDTH/2, self.HEIGHT/2, 60, (255, 255, 100)))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Pygame window for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Space Invaders Gym")
    clock = pygame.time.Clock()

    while running:
        if terminated:
            # Wait 3 seconds then reset
            pygame.time.wait(3000)
            obs, info = env.reset()
            terminated = False

        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(env.FPS)

    env.close()