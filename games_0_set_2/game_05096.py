
# Generated: 2025-08-28T03:56:40.397547
# Source Brief: brief_05096.md
# Brief Index: 5096

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# Set a dummy video driver to run Pygame headlessly
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Hold Shift to cycle weapons. Press Space to fire."
    )

    game_description = (
        "Control a powerful robot in a top-down arena. Strategically destroy all targets before your health runs out."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.ARENA_MARGIN = 20
        self.MAX_STEPS = 1500 # Increased from 1000 for better playability
        self.NUM_TARGETS = 20

        # Colors
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_ARENA = (25, 30, 40)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_TURRET = (150, 220, 255)
        self.COLOR_TARGET = (255, 50, 100)
        self.COLOR_HEALTH_GOOD = (50, 255, 100)
        self.COLOR_HEALTH_BAD = (200, 50, 50)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_EXPLOSION = (255, 200, 50)
        
        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        # Game State (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None
        self.robot = {}
        self.targets = []
        self.projectiles = []
        self.particles = []
        self.weapons = []
        self.current_weapon_idx = 0
        self.fire_cooldown = 0
        self.switch_cooldown = 0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Robot
        self.robot = {
            "pos": np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32),
            "size": 20,
            "speed": 3.0,
            "health": 100,
            "max_health": 100,
            "last_move_dir": np.array([0, -1], dtype=np.float32) # Start aiming up
        }

        # Targets
        self.targets = []
        for _ in range(self.NUM_TARGETS):
            self.targets.append({
                "pos": np.array([
                    self.np_random.uniform(self.ARENA_MARGIN + 20, self.SCREEN_WIDTH - self.ARENA_MARGIN - 20),
                    self.np_random.uniform(self.ARENA_MARGIN + 20, self.SCREEN_HEIGHT - self.ARENA_MARGIN - 20)
                ], dtype=np.float32),
                "radius": 12,
                "health": 50,
                "max_health": 50
            })

        # Weapons
        self.weapons = [
            {"name": "Machine Gun", "ammo": 300, "cooldown": 4, "damage": 6, "speed": 8, "spread": 0.1, "count": 1, "color": (255, 255, 100)},
            {"name": "Shotgun", "ammo": 50, "cooldown": 20, "damage": 15, "speed": 6, "spread": 0.5, "count": 5, "color": (255, 150, 50)},
            {"name": "Laser", "ammo": 20, "cooldown": 30, "damage": 40, "speed": 15, "spread": 0, "count": 1, "color": (100, 255, 255)},
        ]
        self.current_weapon_idx = 0

        # Projectiles and effects
        self.projectiles = []
        self.particles = []

        # Cooldowns
        self.fire_cooldown = 0
        self.switch_cooldown = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.01  # Cost of living

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Update Cooldowns ---
        if self.fire_cooldown > 0: self.fire_cooldown -= 1
        if self.switch_cooldown > 0: self.switch_cooldown -= 1

        # --- Handle Input and Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held, shift_held)

        # --- Update Game Logic ---
        reward += self._update_projectiles()
        self._update_particles()
        
        # --- Check for Termination ---
        self.steps += 1
        terminated = False
        if not self.targets:
            reward += 100  # Victory bonus
            self.game_over = True
            terminated = True
        elif self.robot["health"] <= 0:
            self.game_over = True
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Robot movement
        move_dir = np.array([0, 0], dtype=np.float32)
        if movement == 1: move_dir[1] = -1 # Up
        elif movement == 2: move_dir[1] = 1  # Down
        elif movement == 3: move_dir[0] = -1 # Left
        elif movement == 4: move_dir[0] = 1  # Right
        
        if np.any(move_dir):
            # Normalize for consistent diagonal speed
            norm = np.linalg.norm(move_dir)
            if norm > 0:
                self.robot["pos"] += (move_dir / norm) * self.robot["speed"]
                self.robot["last_move_dir"] = move_dir / norm

        # Clamp robot position to arena
        self.robot["pos"][0] = np.clip(self.robot["pos"][0], self.ARENA_MARGIN + self.robot["size"]/2, self.SCREEN_WIDTH - self.ARENA_MARGIN - self.robot["size"]/2)
        self.robot["pos"][1] = np.clip(self.robot["pos"][1], self.ARENA_MARGIN + self.robot["size"]/2, self.SCREEN_HEIGHT - self.ARENA_MARGIN - self.robot["size"]/2)

        # Cycle weapon
        if shift_held and self.switch_cooldown == 0:
            self.current_weapon_idx = (self.current_weapon_idx + 1) % len(self.weapons)
            self.switch_cooldown = 10 # 1/3 second cooldown
            # Sound: weapon_switch.wav

        # Fire weapon
        weapon = self.weapons[self.current_weapon_idx]
        if space_held and self.fire_cooldown == 0 and weapon["ammo"] > 0:
            self._create_projectiles(weapon)
            weapon["ammo"] -= 1
            self.fire_cooldown = weapon["cooldown"]
            # Sound: mg_fire.wav or shotgun_blast.wav or laser_zap.wav

    def _create_projectiles(self, weapon):
        for i in range(weapon["count"]):
            # Calculate spread
            if weapon["spread"] > 0:
                # For shotgun, create an even spread
                if weapon["count"] > 1:
                    angle_offset = (i - (weapon["count"]-1)/2) * weapon["spread"]
                else: # For MG, create random inaccuracy
                    angle_offset = self.np_random.uniform(-weapon["spread"], weapon["spread"])
                
                base_angle = math.atan2(self.robot["last_move_dir"][1], self.robot["last_move_dir"][0])
                fire_angle = base_angle + angle_offset
                direction = np.array([math.cos(fire_angle), math.sin(fire_angle)], dtype=np.float32)
            else:
                direction = self.robot["last_move_dir"]

            self.projectiles.append({
                "pos": self.robot["pos"].copy(),
                "vel": direction * weapon["speed"],
                "damage": weapon["damage"],
                "color": weapon["color"],
                "lifespan": 60 # Frames
            })

    def _update_projectiles(self):
        hit_reward = 0
        projectiles_to_keep = []
        for p in self.projectiles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            
            hit_target = False
            for target in self.targets:
                dist = np.linalg.norm(p["pos"] - target["pos"])
                if dist < target["radius"]:
                    damage_dealt = min(target["health"], p["damage"])
                    target["health"] -= p["damage"]
                    hit_reward += 0.1 * damage_dealt # Reward for damage
                    self._create_particles(p["pos"], 5, p["color"])
                    hit_target = True
                    # Sound: hit_target.wav
                    break # Projectile hits only one target
            
            if hit_target or p["lifespan"] <= 0 or not (0 < p["pos"][0] < self.SCREEN_WIDTH and 0 < p["pos"][1] < self.SCREEN_HEIGHT):
                continue # Discard projectile
            
            projectiles_to_keep.append(p)
        
        self.projectiles = projectiles_to_keep
        
        # Check for destroyed targets
        targets_to_keep = []
        for target in self.targets:
            if target["health"] > 0:
                targets_to_keep.append(target)
            else:
                hit_reward += 10 # Reward for destroying target
                self.score += 1
                self._create_particles(target["pos"], 20, self.COLOR_EXPLOSION, 1.5)
                # Sound: explosion.wav
        
        self.targets = targets_to_keep
        return hit_reward

    def _create_particles(self, pos, count, color, speed_multiplier=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_multiplier
            self.particles.append({
                "pos": pos.copy(),
                "vel": np.array([math.cos(angle), math.sin(angle)]) * speed,
                "lifespan": self.np_random.integers(10, 20),
                "max_lifespan": 20,
                "color": color,
                "radius": self.np_random.uniform(1, 3)
            })

    def _update_particles(self):
        particles_to_keep = []
        for part in self.particles:
            part["pos"] += part["vel"]
            part["vel"] *= 0.95 # Friction
            part["lifespan"] -= 1
            if part["lifespan"] > 0:
                particles_to_keep.append(part)
        self.particles = particles_to_keep

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Arena
        pygame.draw.rect(self.screen, self.COLOR_ARENA, (self.ARENA_MARGIN, self.ARENA_MARGIN, self.SCREEN_WIDTH - 2*self.ARENA_MARGIN, self.SCREEN_HEIGHT - 2*self.ARENA_MARGIN))

        # Targets
        for target in self.targets:
            pos = target["pos"].astype(int)
            radius = int(target["radius"])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_TARGET)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_TARGET)
            # Health bar
            health_pct = max(0, target["health"] / target["max_health"])
            bar_width = radius * 2
            bar_height = 4
            bar_pos_x = pos[0] - radius
            bar_pos_y = pos[1] - radius - 8
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAD, (bar_pos_x, bar_pos_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_GOOD, (bar_pos_x, bar_pos_y, int(bar_width * health_pct), bar_height))

        # Projectiles
        for p in self.projectiles:
            start = p["pos"] - p["vel"] * 0.5
            end = p["pos"] + p["vel"] * 0.5
            pygame.draw.aaline(self.screen, p["color"], start.astype(int), end.astype(int), 2)

        # Particles
        for part in self.particles:
            life_pct = part["lifespan"] / part["max_lifespan"]
            radius = int(part["radius"] * life_pct)
            if radius > 0:
                pos = part["pos"].astype(int)
                color = tuple(int(c * life_pct) for c in part["color"])
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

        # Robot
        robot_pos = self.robot["pos"].astype(int)
        robot_size = int(self.robot["size"])
        robot_rect = pygame.Rect(robot_pos[0] - robot_size/2, robot_pos[1] - robot_size/2, robot_size, robot_size)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, robot_rect)
        # Turret
        turret_len = robot_size * 0.75
        turret_end = self.robot["pos"] + self.robot["last_move_dir"] * turret_len
        pygame.draw.line(self.screen, self.COLOR_PLAYER_TURRET, robot_pos, turret_end.astype(int), 4)

    def _render_ui(self):
        # Health Bar
        health_pct = max(0, self.robot["health"] / self.robot["max_health"])
        health_bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAD, (10, 10, health_bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GOOD, (10, 10, int(health_bar_width * health_pct), 20))
        health_text = self.font_small.render(f"HEALTH", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 12))
        
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)

        # Weapon Info
        weapon = self.weapons[self.current_weapon_idx]
        weapon_text = self.font_small.render(f"{weapon['name']}: {weapon['ammo']}", True, weapon['color'])
        weapon_rect = weapon_text.get_rect(midbottom=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 10))
        self.screen.blit(weapon_text, weapon_rect)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            if not self.targets:
                end_text = self.font_large.render("VICTORY", True, self.COLOR_HEALTH_GOOD)
            else:
                end_text = self.font_large.render("GAME OVER", True, self.COLOR_TARGET)
            
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.robot["health"],
            "targets_remaining": len(self.targets),
            "ammo": self.weapons[self.current_weapon_idx]["ammo"]
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
    # This block allows you to play the game directly
    # You will need to install pygame for this to work: pip install pygame
    # Re-enable the normal video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Robot Arena")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        # --- Human Input to Action ---
        movement = 0 # none
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
        
        # --- Gym Step ---
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Frame Rate ---
        clock.tick(30) # Match the intended FPS
        
    env.close()