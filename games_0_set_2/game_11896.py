import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:41:31.221797
# Source Brief: brief_01896.md
# Brief Index: 1896
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Leap between lilypads in a strategic top-down shooter. Ambush your enemies while avoiding their attacks to clear the pond."
    )
    user_guide = (
        "Use arrow keys to aim. Press space to leap (stealth mode) or shoot (shooter mode). Press shift to switch between modes."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 2000

    # Colors
    COLOR_BG = (20, 40, 60) # Deep Blue Water
    COLOR_WATER_RIPPLE = (30, 50, 75)
    COLOR_PLAYER = (50, 255, 50) # Bright Green
    COLOR_PLAYER_GLOW = (150, 255, 150)
    COLOR_ENEMY = (255, 50, 50) # Bright Red
    COLOR_PROJECTILE_PLAYER = (255, 255, 0) # Yellow
    COLOR_PROJECTILE_ENEMY = (255, 150, 0) # Orange
    COLOR_LILYPAD_ACTIVE = (0, 150, 80)
    COLOR_LILYPAD_INACTIVE = (40, 80, 60)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)

    # Player settings
    PLAYER_SIZE = 12
    PLAYER_HEALTH_START = 3
    PLAYER_SHOOT_COOLDOWN = 10 # steps
    PLAYER_LEAP_COOLDOWN = 15 # steps
    PLAYER_LEAP_DURATION = 10 # steps

    # Enemy settings
    ENEMY_SIZE = 10
    ENEMY_SHOOT_COOLDOWN = 45 # steps
    ENEMY_SIGHT_RANGE = 250
    ENEMY_ATTACK_RANGE = 200
    ENEMY_PROJECTILE_SPEED_BASE = 3.0

    # Game physics
    PROJECTILE_SIZE = 4
    PROJECTILE_GRAVITY = 0.1
    PARTICLE_LIFESPAN = 20

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # State variables are initialized in reset()
        self.lilypads = []
        self.player = {}
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.ripples = []
        self.zones = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.last_shift_state = 0
        self.enemy_projectile_speed = self.ENEMY_PROJECTILE_SPEED_BASE
        self.enemies_defeated_count = 0

        # self.reset() is called by the wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.victory = False
        self.last_shift_state = 0
        self.enemy_projectile_speed = self.ENEMY_PROJECTILE_SPEED_BASE
        self.enemies_defeated_count = 0

        self._generate_level()
        
        self.player = {
            "pad_idx": 0,
            "pos": np.array(self.lilypads[0]["pos"], dtype=float),
            "health": self.PLAYER_HEALTH_START,
            "mode": "stealth", # 'stealth' or 'shooter'
            "aim_dir": np.array([1.0, 0.0]),
            "shoot_cooldown": 0,
            "leap_cooldown": 0,
            "is_leaping": False,
            "leap_progress": 0,
            "leap_start_pos": None,
            "leap_end_pos": None,
        }

        self.projectiles = []
        self.particles = []
        self.ripples = [self._create_ripple() for _ in range(20)]

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        # --- Update Game Logic based on action ---
        if not self.game_over:
            movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
            
            # Handle state updates
            step_reward = self._update_state(movement, space_pressed, shift_pressed)
            reward += step_reward
            
            # Update all dynamic elements
            self._update_player_leap()
            self._update_enemies()
            self._update_projectiles()
            self._update_particles()
            self._update_ripples()
            
            # Handle collisions and calculate rewards
            collision_reward = self._handle_collisions()
            reward += collision_reward

        self.steps += 1
        
        # --- Check for termination conditions ---
        terminated = False
        truncated = False
        if self.player["health"] <= 0:
            self.game_over = True
            self.victory = False
            reward -= 100
            terminated = True
        
        if len(self.enemies) == 0:
            self.game_over = True
            self.victory = True
            reward += 100
            terminated = True

        if self.steps >= self.MAX_STEPS:
            truncated = True
            terminated = True # Per Gymnasium API, truncated episodes are also terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Private Helper Methods ---

    def _generate_level(self):
        self.lilypads = []
        self.enemies = []
        self.zones = []
        # Use gym's RNG for seeding
        local_random = random.Random(self.np_random.integers(0, 1e6))

        # Zone 0 (Start)
        zone0_pads = [
            {"pos": (100, 200), "radius": 20, "zone": 0, "active": True},
            {"pos": (180, 120), "radius": 15, "zone": 0, "active": True},
            {"pos": (180, 280), "radius": 15, "zone": 0, "active": True},
        ]
        self.lilypads.extend(zone0_pads)
        self.zones.append({"enemy_count": 1, "unlocked": True})
        
        # Zone 1 (Right)
        zone1_pads = [
            {"pos": (320, 80), "radius": 18, "zone": 1, "active": False},
            {"pos": (360, 200), "radius": 22, "zone": 1, "active": False},
            {"pos": (320, 320), "radius": 18, "zone": 1, "active": False},
        ]
        self.lilypads.extend(zone1_pads)
        self.zones.append({"enemy_count": 2, "unlocked": False})

        # Zone 2 (Far Right)
        zone2_pads = [
            {"pos": (500, 150), "radius": 15, "zone": 2, "active": False},
            {"pos": (540, 250), "radius": 20, "zone": 2, "active": False},
            {"pos": (500, 350), "radius": 15, "zone": 2, "active": False},
        ]
        self.lilypads.extend(zone2_pads)
        self.zones.append({"enemy_count": 2, "unlocked": False})

        # Place enemies
        self.enemies.append(self._create_enemy(2, patrol_pads=[1, 2], local_random=local_random)) # Zone 0
        self.enemies.append(self._create_enemy(4, patrol_pads=[3, 4, 5], local_random=local_random)) # Zone 1
        self.enemies.append(self._create_enemy(3, patrol_pads=[3, 5], local_random=local_random)) # Zone 1
        self.enemies.append(self._create_enemy(7, patrol_pads=[6, 7, 8], local_random=local_random)) # Zone 2
        self.enemies.append(self._create_enemy(6, patrol_pads=[6, 8], local_random=local_random)) # Zone 2
    
    def _create_enemy(self, start_pad_idx, patrol_pads, local_random):
        return {
            "pad_idx": start_pad_idx,
            "pos": np.array(self.lilypads[start_pad_idx]["pos"], dtype=float),
            "state": "patrol", # 'patrol', 'attack'
            "shoot_cooldown": local_random.randint(0, self.ENEMY_SHOOT_COOLDOWN),
            "patrol_path": patrol_pads,
            "patrol_idx": 0,
            "is_leaping": False,
            "leap_progress": 0,
            "leap_start_pos": None,
            "leap_end_pos": None,
            "leap_timer": local_random.randint(60, 120)
        }
        
    def _create_ripple(self):
        return {
            "pos": self.np_random.uniform(low=0, high=[self.WIDTH, self.HEIGHT], size=(2,)),
            "radius": 0,
            "max_radius": self.np_random.uniform(20, 60),
            "speed": self.np_random.uniform(0.1, 0.3),
            "alpha": 100,
        }

    def _update_state(self, movement, space_pressed, shift_pressed):
        # Cooldowns
        if self.player["shoot_cooldown"] > 0: self.player["shoot_cooldown"] -= 1
        if self.player["leap_cooldown"] > 0: self.player["leap_cooldown"] -= 1

        # Mode switching (on rising edge of shift)
        if shift_pressed and not self.last_shift_state:
            self.player["mode"] = "shooter" if self.player["mode"] == "stealth" else "stealth"
            # sfx: Mode switch sound
        self.last_shift_state = shift_pressed

        # Aiming
        if movement != 0:
            if movement == 1: self.player["aim_dir"] = np.array([0, -1]) # Up
            elif movement == 2: self.player["aim_dir"] = np.array([0, 1]) # Down
            elif movement == 3: self.player["aim_dir"] = np.array([-1, 0]) # Left
            elif movement == 4: self.player["aim_dir"] = np.array([1, 0]) # Right

        # Action (Space bar)
        if space_pressed:
            if self.player["mode"] == "shooter" and self.player["shoot_cooldown"] == 0 and not self.player["is_leaping"]:
                self._fire_projectile(self.player, self.player["aim_dir"], "player")
                self.player["shoot_cooldown"] = self.PLAYER_SHOOT_COOLDOWN
                # sfx: Player shoot tongue
            
            elif self.player["mode"] == "stealth" and self.player["leap_cooldown"] == 0 and not self.player["is_leaping"]:
                self._initiate_leap(self.player, self.player["aim_dir"])
                self.player["leap_cooldown"] = self.PLAYER_LEAP_COOLDOWN
                # sfx: Player leap whoosh
        
        return 0 # No immediate reward from actions

    def _fire_projectile(self, owner, direction, owner_type):
        start_pos = owner["pos"].copy()
        speed = self.enemy_projectile_speed if owner_type == "enemy" else 7.0
        velocity = direction * speed
        
        self.projectiles.append({
            "pos": start_pos,
            "vel": velocity,
            "owner": owner_type,
            "color": self.COLOR_PROJECTILE_PLAYER if owner_type == "player" else self.COLOR_PROJECTILE_ENEMY
        })
    
    def _initiate_leap(self, entity, direction):
        current_pos = self.lilypads[entity["pad_idx"]]["pos"]
        best_target_idx = -1
        min_dist = float('inf')

        for i, pad in enumerate(self.lilypads):
            if i == entity["pad_idx"] or not pad["active"]:
                continue
            
            vec_to_pad = np.array(pad["pos"]) - current_pos
            dist_to_pad = np.linalg.norm(vec_to_pad)
            if dist_to_pad == 0: continue

            # Check if pad is generally in the right direction
            norm_vec_to_pad = vec_to_pad / dist_to_pad
            dot_product = np.dot(direction, norm_vec_to_pad)
            
            if dot_product > 0.5 and dist_to_pad < min_dist: # Must be somewhat aligned
                min_dist = dist_to_pad
                best_target_idx = i
        
        if best_target_idx != -1:
            entity["is_leaping"] = True
            entity["leap_progress"] = 0
            entity["leap_start_pos"] = entity["pos"].copy()
            entity["leap_end_pos"] = np.array(self.lilypads[best_target_idx]["pos"], dtype=float)
            entity["pad_idx"] = best_target_idx

    def _update_player_leap(self):
        if self.player["is_leaping"]:
            self.player["leap_progress"] += 1
            
            # Ease-out interpolation
            t = self.player["leap_progress"] / self.PLAYER_LEAP_DURATION
            t = 1 - (1 - t)**3 # Cubic ease-out
            
            if self.player["leap_progress"] >= self.PLAYER_LEAP_DURATION:
                self.player["is_leaping"] = False
                self.player["pos"] = self.player["leap_end_pos"].copy()
                self._spawn_particles(self.player["pos"], self.COLOR_WATER_RIPPLE, 5, 1.5)
            else:
                self.player["pos"] = self.player["leap_start_pos"] * (1 - t) + self.player["leap_end_pos"] * t

    def _update_enemies(self):
        for enemy in self.enemies:
            if enemy["is_leaping"]:
                enemy["leap_progress"] += 1
                t = enemy["leap_progress"] / self.PLAYER_LEAP_DURATION
                if enemy["leap_progress"] >= self.PLAYER_LEAP_DURATION:
                    enemy["is_leaping"] = False
                    enemy["pos"] = enemy["leap_end_pos"].copy()
                    self._spawn_particles(enemy["pos"], self.COLOR_WATER_RIPPLE, 3, 1.0)
                else:
                    enemy["pos"] = enemy["leap_start_pos"] * (1 - t) + enemy["leap_end_pos"] * t
                continue

            if enemy["shoot_cooldown"] > 0: enemy["shoot_cooldown"] -= 1
            if enemy["leap_timer"] > 0: enemy["leap_timer"] -= 1

            dist_to_player = np.linalg.norm(self.player["pos"] - enemy["pos"])
            player_is_visible = (self.player["mode"] != 'stealth' or dist_to_player < 80) and not self.player["is_leaping"]

            # FSM: Patrol <-> Attack
            if player_is_visible and dist_to_player < self.ENEMY_SIGHT_RANGE:
                enemy["state"] = "attack"
            else:
                enemy["state"] = "patrol"

            if enemy["state"] == "attack" and dist_to_player < self.ENEMY_ATTACK_RANGE:
                if enemy["shoot_cooldown"] == 0:
                    direction = (self.player["pos"] - enemy["pos"])
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction /= norm
                        self._fire_projectile(enemy, direction, "enemy")
                        enemy["shoot_cooldown"] = self.ENEMY_SHOOT_COOLDOWN
                        # sfx: Enemy shoot tongue
            
            elif enemy["state"] == "patrol":
                if enemy["leap_timer"] == 0:
                    patrol_path = enemy["patrol_path"]
                    enemy["patrol_idx"] = (enemy["patrol_idx"] + 1) % len(patrol_path)
                    target_pad_idx = patrol_path[enemy["patrol_idx"]]
                    
                    direction = np.array(self.lilypads[target_pad_idx]["pos"]) - enemy["pos"]
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction /= norm
                        self._initiate_leap(enemy, direction)
                        enemy["leap_timer"] = self.np_random.integers(90, 181)
    
    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            proj["vel"][1] += self.PROJECTILE_GRAVITY
            proj["pos"] += proj["vel"]
            if not (0 <= proj["pos"][0] < self.WIDTH and 0 <= proj["pos"][1] < self.HEIGHT):
                self.projectiles.remove(proj)

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _update_ripples(self):
        for r in self.ripples:
            r["radius"] += r["speed"]
            if r["radius"] > r["max_radius"]:
                r.update(self._create_ripple())
                r["radius"] = 0 # Start fresh

    def _handle_collisions(self):
        reward = 0
        
        for proj in self.projectiles[:]:
            # Projectile vs Environment (water)
            is_on_lilypad = False
            for pad in self.lilypads:
                if pad["active"] and np.linalg.norm(proj["pos"] - np.array(pad["pos"])) < pad["radius"]:
                    is_on_lilypad = True
                    break
            if not is_on_lilypad and proj in self.projectiles:
                self.projectiles.remove(proj)
                self._spawn_particles(proj["pos"], self.COLOR_WHITE, 5, 2.0)
                # sfx: Splash
                continue

            # Player projectiles vs Enemies
            if proj["owner"] == "player":
                for enemy in self.enemies[:]:
                    if np.linalg.norm(proj["pos"] - enemy["pos"]) < self.ENEMY_SIZE:
                        reward += 0.1 # Hit reward
                        if proj in self.projectiles: self.projectiles.remove(proj)
                        
                        self.enemies.remove(enemy)
                        self.score += 100
                        reward += 1.0 # Kill reward
                        
                        self.enemies_defeated_count += 1
                        if self.enemies_defeated_count % 5 == 0:
                           self.enemy_projectile_speed += 0.05
                        
                        self._spawn_particles(enemy["pos"], self.COLOR_ENEMY, 20, 3.0)
                        # sfx: Enemy defeated explosion
                        
                        # Check for zone unlock
                        zone_idx = self.lilypads[enemy["pad_idx"]]["zone"]
                        self.zones[zone_idx]["enemy_count"] -= 1
                        if self.zones[zone_idx]["enemy_count"] == 0:
                            next_zone = zone_idx + 1
                            if next_zone < len(self.zones) and not self.zones[next_zone]["unlocked"]:
                                self.zones[next_zone]["unlocked"] = True
                                for pad in self.lilypads:
                                    if pad["zone"] == next_zone:
                                        pad["active"] = True
                                reward += 5.0 # Zone unlock reward
                                # sfx: Zone unlocked fanfare
                        break

            # Enemy projectiles vs Player
            elif proj["owner"] == "enemy":
                if not self.player["is_leaping"] and np.linalg.norm(proj["pos"] - self.player["pos"]) < self.PLAYER_SIZE:
                    reward -= 0.1 # Hit penalty
                    self.player["health"] -= 1
                    assert self.player["health"] >= 0
                    if proj in self.projectiles: self.projectiles.remove(proj)
                    self._spawn_particles(self.player["pos"], self.COLOR_PLAYER, 15, 2.5)
                    # sfx: Player hit
                    break
        return reward
        
    def _spawn_particles(self, pos, color, count, speed_scale):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 1.5) * speed_scale
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifespan": self.PARTICLE_LIFESPAN,
                "color": color,
                "size": self.np_random.uniform(1, 4)
            })

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
            "health": self.player["health"],
            "enemies_left": len(self.enemies),
            "player_mode": self.player["mode"],
        }

    def _render_game(self):
        # Draw ripples
        for r in self.ripples:
            alpha = max(0, r["alpha"] * (1 - r["radius"]/r["max_radius"]))
            pygame.gfxdraw.aacircle(self.screen, int(r["pos"][0]), int(r["pos"][1]), int(r["radius"]), (*self.COLOR_WATER_RIPPLE, int(alpha)))

        # Draw lilypads
        for pad in self.lilypads:
            color = self.COLOR_LILYPAD_ACTIVE if pad["active"] else self.COLOR_LILYPAD_INACTIVE
            pos_i = (int(pad["pos"][0]), int(pad["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos_i[0], pos_i[1], pad["radius"], color)
            pygame.gfxdraw.aacircle(self.screen, pos_i[0], pos_i[1], pad["radius"], tuple(int(c*0.8) for c in color))
        
        # Draw aim/leap indicator
        if not self.player["is_leaping"]:
            start_pos = self.player["pos"]
            end_pos = start_pos + self.player["aim_dir"] * 40
            pygame.draw.line(self.screen, self.COLOR_WHITE, start_pos.astype(int), end_pos.astype(int), 1)

        # Draw projectiles
        for proj in self.projectiles:
            pos_i = proj["pos"].astype(int)
            pygame.gfxdraw.filled_circle(self.screen, pos_i[0], pos_i[1], self.PROJECTILE_SIZE, proj["color"])
            pygame.gfxdraw.aacircle(self.screen, pos_i[0], pos_i[1], self.PROJECTILE_SIZE, self.COLOR_WHITE)

        # Draw enemies
        for enemy in self.enemies:
            self._draw_frog(self.screen, enemy["pos"], self.COLOR_ENEMY, self.ENEMY_SIZE, False, np.arctan2(*(self.player["pos"]-enemy["pos"])[::-1]))

        # Draw player
        self._draw_frog(self.screen, self.player["pos"], self.COLOR_PLAYER, self.PLAYER_SIZE, self.player["mode"] == 'stealth', np.arctan2(*self.player["aim_dir"][::-1]), True)
        self._draw_player_hud()

        # Draw particles
        for p in self.particles:
            alpha = 255 * (p["lifespan"] / self.PARTICLE_LIFESPAN)
            color_with_alpha = (*p["color"], int(alpha))
            pos_i = p["pos"].astype(int)
            radius = int(p["size"] * (p["lifespan"] / self.PARTICLE_LIFESPAN))
            # Ensure radius is drawable and position is a tuple to avoid numpy ambiguity
            if radius > 0:
                pygame.draw.circle(self.screen, color_with_alpha, tuple(pos_i), radius)


    def _draw_frog(self, surface, pos, color, size, is_stealth, angle, is_player=False):
        pos_i = pos.astype(int)
        
        if is_player and not self.player["is_leaping"]:
             # Glow effect
            glow_alpha = 100 if not is_stealth else 40
            pygame.gfxdraw.filled_circle(surface, pos_i[0], pos_i[1], int(size * 1.8), (*self.COLOR_PLAYER_GLOW, glow_alpha))
            pygame.gfxdraw.aacircle(surface, pos_i[0], pos_i[1], int(size * 1.8), (*self.COLOR_PLAYER_GLOW, glow_alpha))

        # Body
        body_alpha = 128 if is_stealth else 255
        pygame.gfxdraw.filled_circle(surface, pos_i[0], pos_i[1], size, (*color, body_alpha))
        pygame.gfxdraw.aacircle(surface, pos_i[0], pos_i[1], size, (*color, body_alpha))
        
        # Eyes
        eye_offset = size * 0.8
        eye_pos1 = (pos_i[0] + eye_offset * math.cos(angle - 0.5), pos_i[1] + eye_offset * math.sin(angle - 0.5))
        eye_pos2 = (pos_i[0] + eye_offset * math.cos(angle + 0.5), pos_i[1] + eye_offset * math.sin(angle + 0.5))
        eye_size = int(size * 0.4)
        pygame.gfxdraw.filled_circle(surface, int(eye_pos1[0]), int(eye_pos1[1]), eye_size, self.COLOR_WHITE)
        pygame.gfxdraw.filled_circle(surface, int(eye_pos2[0]), int(eye_pos2[1]), eye_size, self.COLOR_WHITE)
        pygame.gfxdraw.filled_circle(surface, int(eye_pos1[0]), int(eye_pos1[1]), eye_size-1, self.COLOR_BLACK)
        pygame.gfxdraw.filled_circle(surface, int(eye_pos2[0]), int(eye_pos2[1]), eye_size-1, self.COLOR_BLACK)

    def _draw_player_hud(self):
        hud_pos = self.player["pos"] - [0, self.PLAYER_SIZE + 15]
        
        # Health hearts
        for i in range(self.player["health"]):
            heart_pos = (int(hud_pos[0] - 15 + i*12), int(hud_pos[1]))
            pygame.draw.polygon(self.screen, self.COLOR_ENEMY, [(heart_pos[0], heart_pos[1]+2), (heart_pos[0]+4, heart_pos[1]-2), (heart_pos[0]+8, heart_pos[1]+2), (heart_pos[0]+4, heart_pos[1]+6)])

        # Mode icon
        icon_pos = (int(hud_pos[0] + 25), int(hud_pos[1]))
        if self.player["mode"] == 'stealth':
            # Draw an eye
            pygame.draw.ellipse(self.screen, self.COLOR_WHITE, (icon_pos[0], icon_pos[1], 12, 8))
            pygame.draw.circle(self.screen, self.COLOR_BG, (icon_pos[0]+6, icon_pos[1]+4), 2)
        else: # shooter
            # Draw a tongue
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE_PLAYER, icon_pos, (icon_pos[0]+10, icon_pos[1]+5), 2)

    def _render_ui(self):
        # Enemy Count
        enemy_text = self.font_ui.render(f"ENEMIES: {len(self.enemies)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(enemy_text, (self.WIDTH - enemy_text.get_width() - 10, 10))
        
        # Game Over / Victory
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0,0,0, 180))
            self.screen.blit(overlay, (0,0))
            
            msg = "VICTORY!" if self.victory else "GAME OVER"
            color = self.COLOR_PLAYER if self.victory else self.COLOR_ENEMY
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

# --- Example Usage ---
if __name__ == '__main__':
    env = GameEnv()
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    # Re-enable video driver for local play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    pygame.display.set_caption("Frogger's Leap")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    movement = 0
    space_held = 0
    shift_held = 0
    
    terminated, truncated = False, False
    while not (terminated or truncated):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            
            # Handle key presses for manual control
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: space_held = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE: space_held = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 0

        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}, Steps: {info['steps']}")
            # obs, info = env.reset() # This would start a new episode

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)
        
    pygame.quit()