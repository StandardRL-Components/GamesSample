import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:06:25.957382
# Source Brief: brief_00269.md
# Brief Index: 269
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    game_description = (
        "Play as a cyber ninja defeating guards in a top-down stealth action game. "
        "Match your blade color to your enemies for extra damage and clear the area to win."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to attack and shift to cycle your blade color."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 2000
        self.N_INITIAL_GUARDS = 4

        # Colors
        self.COLOR_BG = (10, 0, 20)
        self.COLOR_WALL = (40, 20, 60)
        self.COLOR_PLAYER = (220, 255, 255)
        self.COLOR_PLAYER_GLOW = (100, 180, 255)
        self.BLADE_COLORS = {
            "red": (255, 50, 50),
            "green": (50, 255, 50),
            "blue": (50, 100, 255),
            "gray": (150, 150, 150)
        }
        self.BLADE_COLOR_ORDER = ["red", "green", "blue", "gray"]
        self.UI_TEXT_COLOR = (220, 220, 220)
        self.UI_HEALTH_COLOR = (200, 40, 40)
        self.UI_HEALTH_BG_COLOR = (50, 10, 10)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player = None
        self.guards = []
        self.walls = []
        self.particles = []
        self.attacks = []
        self.space_pressed_last_frame = False
        self.shift_pressed_last_frame = False
        
        # self.reset() # reset is called by the environment wrapper
        # self.validate_implementation() # this is for debugging and should not be in __init__
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Player state
        self.player = {
            "pos": pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2),
            "radius": 10,
            "speed": 4,
            "health": 100,
            "max_health": 100,
            "blade_color_idx": 0,
            "attack_cooldown": 0,
            "invulnerable_timer": 0,
        }
        
        self.guards = []
        self.walls = self._generate_walls()
        self._generate_guards()

        self.particles = []
        self.attacks = []
        self.space_pressed_last_frame = True # Prevent action on first frame
        self.shift_pressed_last_frame = True # Prevent action on first frame

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        # Handle input and state updates
        reward += self._handle_input(movement, space_held, shift_held)
        self._update_player()
        self._update_guards()
        reward += self._update_attacks_and_collisions()
        self._update_particles()
        
        # Update game state
        self.steps += 1
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated and not self.game_over:
            if not self.guards:
                reward += 100 # Victory bonus
            # No death penalty, accumulated negative rewards suffice
            self.game_over = True
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Movement
        target_pos = self.player["pos"].copy()
        if movement == 1: target_pos.y -= self.player["speed"] # Up
        elif movement == 2: target_pos.y += self.player["speed"] # Down
        elif movement == 3: target_pos.x -= self.player["speed"] # Left
        elif movement == 4: target_pos.x += self.player["speed"] # Right
        
        # Wall collision for movement
        potential_rect = pygame.Rect(target_pos.x - self.player["radius"], target_pos.y - self.player["radius"], self.player["radius"]*2, self.player["radius"]*2)
        collided = False
        for wall in self.walls:
            if wall.colliderect(potential_rect):
                collided = True
                break
        if not collided:
            self.player["pos"] = target_pos

        # Boundary checks
        self.player["pos"].x = np.clip(self.player["pos"].x, self.player["radius"], self.WIDTH - self.player["radius"])
        self.player["pos"].y = np.clip(self.player["pos"].y, self.player["radius"], self.HEIGHT - self.player["radius"])

        # Action: Attack (on key press)
        if space_held and not self.space_pressed_last_frame and self.player["attack_cooldown"] <= 0:
            self.player["attack_cooldown"] = 15 # 0.5s cooldown at 30fps
            blade_color_name = self.BLADE_COLOR_ORDER[self.player["blade_color_idx"]]
            self.attacks.append({
                "pos": self.player["pos"].copy(),
                "angle": self.np_random.uniform(0, 2 * math.pi),
                "radius": 40,
                "lifetime": 5, # 5 frames
                "color": self.BLADE_COLORS[blade_color_name],
                "color_name": blade_color_name
            })
        self.space_pressed_last_frame = space_held

        # Action: Cycle blade color (on key press)
        if shift_held and not self.shift_pressed_last_frame:
            self.player["blade_color_idx"] = (self.player["blade_color_idx"] + 1) % len(self.BLADE_COLOR_ORDER)
        self.shift_pressed_last_frame = shift_held

        return 0.0 # Input handling itself gives no reward

    def _update_player(self):
        if self.player["attack_cooldown"] > 0:
            self.player["attack_cooldown"] -= 1
        if self.player["invulnerable_timer"] > 0:
            self.player["invulnerable_timer"] -= 1

    def _update_guards(self):
        difficulty_speed_mod = 1.0 + (0.05 * (self.steps // 500))
        reward = 0

        for guard in self.guards:
            target_point = guard["path"][guard["path_idx"]]
            direction = (target_point - guard["pos"])
            
            if direction.length() < guard["speed"] * difficulty_speed_mod:
                guard["pos"] = target_point
                guard["path_idx"] = (guard["path_idx"] + 1) % len(guard["path"])
            else:
                guard["pos"] += direction.normalize() * guard["speed"] * difficulty_speed_mod
            
            # Check collision with player
            if self.player["invulnerable_timer"] <= 0:
                dist = self.player["pos"].distance_to(guard["pos"])
                if dist < self.player["radius"] + guard["radius"]:
                    self.player["health"] -= 10
                    self.player["invulnerable_timer"] = 30 # 1s invulnerability
                    reward -= 0.1 # Damage penalty
                    self._create_particles(self.player["pos"], self.COLOR_PLAYER, 20, 3)
        return reward

    def _update_attacks_and_collisions(self):
        reward = 0
        for attack in self.attacks[:]:
            attack["lifetime"] -= 1
            if attack["lifetime"] <= 0:
                self.attacks.remove(attack)
                continue

            for guard in self.guards[:]:
                dist = attack["pos"].distance_to(guard["pos"])
                if dist < attack["radius"] + guard["radius"]:
                    is_color_match = attack["color_name"] == guard["color_name"]
                    damage = 20 if is_color_match else 10
                    guard["health"] -= damage
                    
                    reward += 0.1 # Reward for any damage
                    self._create_particles(guard["pos"], guard["color"], 15, 2, is_color_match)

                    if guard["health"] <= 0:
                        reward += 2.0 if is_color_match else 1.0
                        self._create_particles(guard["pos"], guard["color"], 50, 4)
                        self.guards.remove(guard)
                    
                    # Attack is consumed on first hit
                    if attack in self.attacks:
                        self.attacks.remove(attack)
                    break
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # Damping
            p["lifetime"] -= 1
            if p["lifetime"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.player["health"] <= 0:
            return True
        if not self.guards:
            return True
        # Truncation is handled separately
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render walls
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)
        
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p["lifetime"] / p["max_lifetime"]))
            color = p["color"] + (alpha,)
            temp_surf = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(temp_surf, (int(p["pos"].x - p["radius"]), int(p["pos"].y - p["radius"])))

        # Render guards
        for guard in self.guards:
            pos = (int(guard["pos"].x), int(guard["pos"].y))
            pygame.draw.circle(self.screen, guard["color"], pos, guard["radius"])
            pygame.draw.circle(self.screen, (0,0,0), pos, guard["radius"], 2)
            # Health bar
            if guard["health"] < guard["max_health"]:
                hb_width = 30
                hb_height = 5
                hb_x = guard["pos"].x - hb_width / 2
                hb_y = guard["pos"].y - guard["radius"] - 10
                fill_ratio = guard["health"] / guard["max_health"]
                pygame.draw.rect(self.screen, self.UI_HEALTH_BG_COLOR, (hb_x, hb_y, hb_width, hb_height))
                pygame.draw.rect(self.screen, self.UI_HEALTH_COLOR, (hb_x, hb_y, hb_width * fill_ratio, hb_height))

        # Render attacks
        for attack in self.attacks:
            alpha = int(255 * (attack["lifetime"] / 5))
            color = attack["color"]
            temp_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            pygame.gfxdraw.arc(temp_surf, int(attack["pos"].x), int(attack["pos"].y), attack["radius"], int(math.degrees(attack["angle"])), int(math.degrees(attack["angle"]) + 120), color + (alpha,))
            pygame.gfxdraw.arc(temp_surf, int(attack["pos"].x), int(attack["pos"].y), attack["radius"]-1, int(math.degrees(attack["angle"])), int(math.degrees(attack["angle"]) + 120), color + (alpha,))
            pygame.gfxdraw.arc(temp_surf, int(attack["pos"].x), int(attack["pos"].y), attack["radius"]-2, int(math.degrees(attack["angle"])), int(math.degrees(attack["angle"]) + 120), color + (alpha,))
            self.screen.blit(temp_surf, (0, 0))

        # Render player
        player_pos = (int(self.player["pos"].x), int(self.player["pos"].y))
        # Invulnerability flash
        if self.player["invulnerable_timer"] > 0 and self.steps % 4 < 2:
            pass # Don't render player to make it flash
        else:
            # Glow effect
            glow_radius = int(self.player["radius"] * 1.8)
            glow_alpha = 80
            temp_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, self.COLOR_PLAYER_GLOW + (glow_alpha,), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(temp_surf, (player_pos[0] - glow_radius, player_pos[1] - glow_radius))

            # Player body
            pygame.draw.circle(self.screen, self.COLOR_PLAYER, player_pos, self.player["radius"])
            pygame.draw.circle(self.screen, self.COLOR_BG, player_pos, self.player["radius"], 2)

    def _render_ui(self):
        # Health Bar
        hb_width = 150
        hb_height = 20
        fill_ratio = max(0, self.player["health"] / self.player["max_health"])
        pygame.draw.rect(self.screen, self.UI_HEALTH_BG_COLOR, (10, 10, hb_width, hb_height))
        pygame.draw.rect(self.screen, self.UI_HEALTH_COLOR, (10, 10, hb_width * fill_ratio, hb_height))
        pygame.draw.rect(self.screen, self.UI_TEXT_COLOR, (10, 10, hb_width, hb_height), 1)
        health_text = self.font_small.render(f"HEALTH", True, self.UI_TEXT_COLOR)
        self.screen.blit(health_text, (15, 12))

        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.UI_TEXT_COLOR)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Blade Color Indicator
        blade_color_name = self.BLADE_COLOR_ORDER[self.player["blade_color_idx"]]
        blade_color = self.BLADE_COLORS[blade_color_name]
        indicator_rect = pygame.Rect(self.WIDTH - 60, self.HEIGHT - 60, 50, 50)
        pygame.draw.rect(self.screen, blade_color, indicator_rect)
        pygame.draw.rect(self.screen, self.UI_TEXT_COLOR, indicator_rect, 2)
        blade_text = self.font_small.render("BLADE", True, self.UI_TEXT_COLOR)
        self.screen.blit(blade_text, (indicator_rect.x, indicator_rect.y - 20))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player["health"],
            "guards_remaining": len(self.guards),
        }

    def _generate_walls(self):
        walls = []
        # Create a border
        walls.append(pygame.Rect(0, 0, self.WIDTH, 5))
        walls.append(pygame.Rect(0, 0, 5, self.HEIGHT))
        walls.append(pygame.Rect(self.WIDTH - 5, 0, 5, self.HEIGHT))
        walls.append(pygame.Rect(0, self.HEIGHT - 5, self.WIDTH, 5))
        
        # Add some random internal walls
        for _ in range(self.np_random.integers(3, 6)):
            is_vertical = self.np_random.choice([True, False])
            if is_vertical:
                w, h = 15, self.np_random.integers(50, 150)
            else:
                w, h = self.np_random.integers(50, 150), 15
            
            x = self.np_random.integers(50, self.WIDTH - 50 - w)
            y = self.np_random.integers(50, self.HEIGHT - 50 - h)
            walls.append(pygame.Rect(x, y, w, h))
        return walls

    def _generate_guards(self):
        for _ in range(self.N_INITIAL_GUARDS):
            color_name = self.np_random.choice(self.BLADE_COLOR_ORDER[:3]) # No gray guards
            color = self.BLADE_COLORS[color_name]
            
            # Generate a patrol path
            path = []
            for _ in range(self.np_random.integers(2, 5)):
                is_valid = False
                while not is_valid:
                    point = pygame.Vector2(
                        self.np_random.integers(30, self.WIDTH - 30),
                        self.np_random.integers(30, self.HEIGHT - 30)
                    )
                    point_rect = pygame.Rect(point.x-10, point.y-10, 20, 20)
                    if not any(wall.colliderect(point_rect) for wall in self.walls):
                        is_valid = True
                path.append(point)

            self.guards.append({
                "pos": path[0].copy(),
                "radius": 8,
                "speed": self.np_random.uniform(1.0, 1.5),
                "health": 40,
                "max_health": 40,
                "color": color,
                "color_name": color_name,
                "path": path,
                "path_idx": 0,
            })

    def _create_particles(self, pos, color, count, speed_scale, is_special=False):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_scale
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifetime = self.np_random.integers(10, 25)
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifetime": lifetime,
                "max_lifetime": lifetime,
                "color": self.BLADE_COLORS["red"] if is_special else color,
                "radius": self.np_random.integers(1, 4) if not is_special else self.np_random.integers(3, 6)
            })

    def close(self):
        pygame.quit()

    def render(self):
        # The render method is required by the Gym API for render_mode="rgb_array"
        return self._get_observation()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # Make sure to re-enable the display for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Cyber Ninja Stealth")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    while not done:
        # --- Human Input ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # --- Pygame Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(env.metadata["render_fps"])

    print(f"Game Over! Final Score: {total_reward:.2f}")
    env.close()