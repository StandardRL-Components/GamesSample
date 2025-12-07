
# Generated: 2025-08-28T06:26:01.254651
# Source Brief: brief_02935.md
# Brief Index: 2935

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
        "Controls: ←→ to move, ↑ to jump. Press space to fire a projectile."
    )

    game_description = (
        "A fast-paced, side-scrolling arcade shooter. Defeat waves of monsters to score points. "
        "Each new wave brings faster enemies. Survive as long as you can!"
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame Setup
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 20)
        self.font_large = pygame.font.SysFont("Consolas", 40, bold=True)

        # Game Constants
        self.GROUND_Y = 350
        self.GRAVITY = 0.8
        self.MAX_STEPS = 2000

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GROUND = (50, 60, 80)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_GLOW = (50, 255, 150, 50)
        self.COLOR_PROJECTILE = (255, 200, 50)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_SHADOW = (10, 10, 10)
        self.MONSTER_COLORS = {
            "walker": (255, 80, 80),
            "jumper": (200, 100, 255),
        }
        self.PARTICLE_COLORS = {
            "hit": (255, 150, 50),
            "death": (200, 200, 200),
        }
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.wave = 1
        self.game_over = False
        self.monster_base_speed = 0.0
        self.player = {}
        self.monsters = []
        self.projectiles = []
        self.particles = []
        self.prev_space_held = False
        self.screen_shake = 0
        
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.wave = 1
        self.game_over = False
        self.monster_base_speed = 1.0
        
        self.player = {
            "pos": np.array([self.WIDTH / 2, self.GROUND_Y], dtype=np.float64),
            "vel": np.array([0.0, 0.0], dtype=np.float64),
            "size": np.array([20, 40]),
            "health": 50,
            "max_health": 50,
            "on_ground": True,
            "attack_cooldown": 0,
            "iframes": 0, # Invincibility frames
        }
        
        self.monsters = []
        self.projectiles = []
        self.particles = []
        self.prev_space_held = False
        self.screen_shake = 0

        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if not self.game_over:
            # Handle Input and Cooldowns
            self._handle_input(movement, space_held)
            self.player["attack_cooldown"] = max(0, self.player["attack_cooldown"] - 1)
            self.player["iframes"] = max(0, self.player["iframes"] - 1)
            self.screen_shake = max(0, self.screen_shake - 1)

            # Update Game Logic
            self._update_player()
            self._update_monsters()
            self._update_projectiles()
            
            # Handle Collisions and Get Rewards
            dmg_dealt, dmg_taken = self._handle_collisions()
            reward += dmg_dealt * 0.1
            reward -= dmg_taken * 0.1
            
            # Clean up dead monsters
            monsters_killed = len([m for m in self.monsters if m["health"] <= 0])
            if monsters_killed > 0:
                reward += monsters_killed * 10
                self.score += monsters_killed * 100
                self.monsters = [m for m in self.monsters if m["health"] > 0]
            
            # Check for wave completion
            if not self.monsters:
                reward += 100
                self.score += 500
                self.wave += 1
                self.monster_base_speed += 0.2
                self._spawn_wave()
                # sound: wave_complete.wav
        
        # Update particles regardless of game over state for lingering effects
        self._update_particles()
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated and not self.game_over:
            reward -= 100
            self.game_over = True
            # sound: game_over.wav
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        render_offset = (0,0)
        if self.screen_shake > 0:
            render_offset = (self.np_random.integers(-5, 6), self.np_random.integers(-5, 6))

        # Create a temporary surface to apply the shake
        temp_surf = self.screen.copy()
        temp_surf.fill(self.COLOR_BG)
        
        self._render_background(temp_surf)
        self._render_particles(temp_surf)
        self._render_monsters(temp_surf)
        self._render_player(temp_surf)
        self._render_projectiles(temp_surf)
        
        # Blit the shaken surface to the main screen
        self.screen.fill(self.COLOR_BG)
        self.screen.blit(temp_surf, render_offset)

        self._render_ui(self.screen)
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "player_health": self.player["health"],
            "monsters_left": len(self.monsters),
        }

    # --- Helper Methods ---

    def _handle_input(self, movement, space_held):
        # Horizontal Movement
        if movement == 3: # Left
            self.player["vel"][0] -= 1.2
        elif movement == 4: # Right
            self.player["vel"][0] += 1.2
        
        # Apply friction
        self.player["vel"][0] *= 0.85
        self.player["vel"][0] = np.clip(self.player["vel"][0], -7, 7)

        # Jumping
        if movement == 1 and self.player["on_ground"]:
            self.player["vel"][1] = -15
            self.player["on_ground"] = False
            # sound: jump.wav

        # Attacking
        if space_held and not self.prev_space_held and self.player["attack_cooldown"] == 0:
            direction = 1 if self.player["vel"][0] >= 0 else -1
            proj_pos = self.player["pos"] + np.array([self.player["size"][0]/2 * direction, -self.player["size"][1]/2])
            proj_vel = np.array([15.0 * direction, 0.0])
            self.projectiles.append({"pos": proj_pos, "vel": proj_vel, "size": 8, "lifespan": 40})
            self.player["attack_cooldown"] = 15 # 0.5s at 30fps
            # sound: shoot.wav

        self.prev_space_held = space_held

    def _update_player(self):
        # Apply gravity
        if not self.player["on_ground"]:
            self.player["vel"][1] += self.GRAVITY
        
        # Update position
        self.player["pos"] += self.player["vel"]
        
        # Ground collision
        if self.player["pos"][1] > self.GROUND_Y:
            self.player["pos"][1] = self.GROUND_Y
            self.player["vel"][1] = 0
            self.player["on_ground"] = True

        # Screen bounds
        self.player["pos"][0] = np.clip(self.player["pos"][0], 0, self.WIDTH - self.player["size"][0])

    def _update_monsters(self):
        for m in self.monsters:
            m["iframes"] = max(0, m["iframes"] - 1)
            
            if m["type"] == "walker":
                m["pos"][0] += m["vel"][0]
                if m["pos"][0] <= m["bounds"][0] or m["pos"][0] >= m["bounds"][1]:
                    m["vel"][0] *= -1
            
            elif m["type"] == "jumper":
                if m["on_ground"]:
                    m["jump_cooldown"] -= 1
                    if m["jump_cooldown"] <= 0:
                        m["vel"][1] = -self.np_random.uniform(8, 14)
                        m["on_ground"] = False
                else:
                    m["vel"][1] += self.GRAVITY
                
                m["pos"] += m["vel"]

                if m["pos"][1] > self.GROUND_Y:
                    m["pos"][1] = self.GROUND_Y
                    m["vel"][1] = 0
                    m["on_ground"] = True
                    m["jump_cooldown"] = self.np_random.integers(30, 90)

    def _update_projectiles(self):
        for p in self.projectiles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
        self.projectiles = [p for p in self.projectiles if p["lifespan"] > 0 and 0 < p["pos"][0] < self.WIDTH]

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"][1] += 0.1 # particle gravity
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _handle_collisions(self):
        damage_dealt = 0
        damage_taken = 0
        
        player_rect = pygame.Rect(self.player["pos"] - np.array([0, self.player["size"][1]]), self.player["size"])

        # Projectile vs Monster
        for p in self.projectiles[:]:
            proj_rect = pygame.Rect(p["pos"] - p["size"]/2, (p["size"], p["size"]))
            for m in self.monsters:
                if m["iframes"] > 0: continue
                monster_rect = pygame.Rect(m["pos"] - np.array([0, m["size"][1]]), m["size"])
                if proj_rect.colliderect(monster_rect):
                    dmg = self.np_random.integers(3, 6)
                    m["health"] -= dmg
                    damage_dealt += dmg
                    m["iframes"] = 10
                    self._spawn_particles(p["pos"][0], p["pos"][1], self.PARTICLE_COLORS["hit"], 15)
                    if m["health"] <= 0:
                        self._spawn_particles(m["pos"][0], m["pos"][1] - m["size"][1]/2, self.PARTICLE_COLORS["death"], 40)
                        # sound: monster_die.wav
                    else:
                        # sound: monster_hit.wav
                        pass
                    if p in self.projectiles: self.projectiles.remove(p)
                    break

        # Player vs Monster
        if self.player["iframes"] == 0:
            for m in self.monsters:
                monster_rect = pygame.Rect(m["pos"] - np.array([0, m["size"][1]]), m["size"])
                if player_rect.colliderect(monster_rect):
                    dmg = self.np_random.integers(5, 11)
                    self.player["health"] -= dmg
                    damage_taken += dmg
                    self.player["iframes"] = 60 # 2 seconds of invincibility
                    self.screen_shake = 10
                    # sound: player_hit.wav
                    break
        
        return damage_dealt, damage_taken

    def _spawn_wave(self):
        for _ in range(7):
            monster_type = self.np_random.choice(["walker", "jumper"])
            x_pos = self.np_random.uniform(50, self.WIDTH - 50)
            
            if monster_type == "walker":
                size = np.array([30, 30])
                speed = self.np_random.uniform(0.8, 1.2) * self.monster_base_speed
                bound_range = self.np_random.uniform(50, 150)
                bounds = (max(0, x_pos - bound_range), min(self.WIDTH - size[0], x_pos + bound_range))
                self.monsters.append({
                    "pos": np.array([x_pos, self.GROUND_Y], dtype=np.float64),
                    "vel": np.array([speed, 0.0]),
                    "size": size, "health": 10, "type": "walker", "iframes": 0, "bounds": bounds
                })
            elif monster_type == "jumper":
                size = np.array([25, 25])
                self.monsters.append({
                    "pos": np.array([x_pos, self.GROUND_Y], dtype=np.float64),
                    "vel": np.array([0.0, 0.0]),
                    "size": size, "health": 10, "type": "jumper", "iframes": 0,
                    "on_ground": True, "jump_cooldown": self.np_random.integers(30, 90)
                })

    def _spawn_particles(self, x, y, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                "pos": np.array([x, y], dtype=np.float64),
                "vel": vel,
                "lifespan": self.np_random.integers(15, 30),
                "color": color,
                "size": self.np_random.uniform(2, 5)
            })

    def _check_termination(self):
        return self.player["health"] <= 0 or self.steps >= self.MAX_STEPS

    # --- Rendering Methods ---

    def _render_background(self, surface):
        surface.fill(self.COLOR_BG)
        pygame.draw.rect(surface, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))

    def _render_player(self, surface):
        # Player Glow
        glow_radius = int(self.player["size"][1] * 0.8)
        glow_center = (
            int(self.player["pos"][0] + self.player["size"][0]/2), 
            int(self.player["pos"][1] - self.player["size"][1]/2)
        )
        s = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        pygame.draw.circle(s, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        surface.blit(s, (glow_center[0] - glow_radius, glow_center[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Player Body
        rect = pygame.Rect(
            int(self.player["pos"][0]), 
            int(self.player["pos"][1] - self.player["size"][1]), 
            int(self.player["size"][0]), 
            int(self.player["size"][1])
        )
        
        color = self.COLOR_PLAYER
        if self.player["iframes"] > 0 and self.steps % 4 < 2:
            color = (255, 255, 255) # Flash white when invincible

        pygame.draw.rect(surface, color, rect, border_radius=3)

    def _render_monsters(self, surface):
        for m in self.monsters:
            rect = pygame.Rect(
                int(m["pos"][0]), 
                int(m["pos"][1] - m["size"][1]), 
                int(m["size"][0]), 
                int(m["size"][1])
            )
            color = self.MONSTER_COLORS[m["type"]]
            if m["iframes"] > 0:
                color = (255, 255, 255) # Flash white when hit
            
            if m["type"] == "walker":
                pygame.draw.rect(surface, color, rect, border_radius=2)
            elif m["type"] == "jumper":
                pygame.draw.ellipse(surface, color, rect)

    def _render_projectiles(self, surface):
        for p in self.projectiles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            pygame.draw.circle(surface, self.COLOR_PROJECTILE, pos, int(p["size"]))
            pygame.gfxdraw.aacircle(surface, pos[0], pos[1], int(p["size"]), self.COLOR_PROJECTILE)

    def _render_particles(self, surface):
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 30))
            color = (*p["color"], alpha)
            s = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p["size"], p["size"]), p["size"])
            surface.blit(s, (p["pos"][0]-p["size"], p["pos"][1]-p["size"]), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self, surface):
        # Health Bar
        health_ratio = max(0, self.player["health"] / self.player["max_health"])
        health_bar_width = 200
        health_bar_rect = pygame.Rect(10, 10, int(health_bar_width * health_ratio), 20)
        pygame.draw.rect(surface, (150, 0, 0), (10, 10, health_bar_width, 20))
        pygame.draw.rect(surface, (0, 200, 0), health_bar_rect)
        pygame.draw.rect(surface, self.COLOR_TEXT, (10, 10, health_bar_width, 20), 1)

        # Score and Wave
        self._draw_text(f"SCORE: {self.score}", (self.WIDTH - 10, 10), self.COLOR_TEXT, self.font_small, "topright")
        self._draw_text(f"WAVE: {self.wave}", (self.WIDTH / 2, 10), self.COLOR_TEXT, self.font_small, "midtop")
        
        if self.game_over:
            self._draw_text("GAME OVER", (self.WIDTH/2, self.HEIGHT/2 - 30), (255, 50, 50), self.font_large, "center")

    def _draw_text(self, text, pos, color, font, align="topleft"):
        shadow_surf = font.render(text, True, self.COLOR_SHADOW)
        text_surf = font.render(text, True, color)
        
        shadow_rect = shadow_surf.get_rect()
        text_rect = text_surf.get_rect()

        setattr(text_rect, align, pos)
        setattr(shadow_rect, align, (pos[0] + 2, pos[1] + 2))
        
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To run and play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen_width, screen_height = 640, 400
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Monster Wave Defender")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(env.user_guide)

    while running:
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Convert observation for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(30) # Match the intended FPS
        
    pygame.quit()