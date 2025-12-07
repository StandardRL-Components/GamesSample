import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:31:17.861027
# Source Brief: brief_01113.md
# Brief Index: 1113
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    Gymnasium environment for a cyberpunk-themed tower defense game.
    The agent controls an aiming reticle to defend a central core from
    invading viruses by launching data packets.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your system's core from invading viruses by aiming a reticle and launching data packets."
    )
    user_guide = (
        "Controls: Use ↑↓ arrow keys to adjust the firing angle and ←→ to adjust the launch power. "
        "Press space to fire a data packet."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 5000
    TARGET_FPS = 30 # For visual smoothness, not physics

    # --- Colors (Cyberpunk Palette) ---
    COLOR_BG = (10, 20, 40)
    COLOR_GRID = (20, 40, 70)
    COLOR_CORE = (0, 255, 255)
    COLOR_CORE_DANGER = (255, 100, 0)
    COLOR_PLAYER_RETICLE = (255, 255, 0)
    COLOR_PACKET = (100, 255, 100)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_HEADLINE = (0, 255, 255)

    VIRUS_COLORS = {
        "RED": (255, 50, 50),
        "BLUE": (50, 150, 255),
        "PURPLE": (200, 50, 255),
        "YELLOW": (255, 255, 0),
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.system_health = 0
        self.viruses_destroyed = 0
        self.level = 1
        self.aim_angle = 0.0
        self.launch_power = 0.0
        self.prev_space_held = False
        self.core_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=float)
        
        self.packets = []
        self.viruses = []
        self.particles = []

        self.virus_spawn_timer = 0.0
        
        # --- Run validation check ---
        # self.validate_implementation() # Commented out for production


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Initialize Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.system_health = 100.0
        self.viruses_destroyed = 0
        self.level = 1
        
        self.aim_angle = -math.pi / 2  # Start pointing up
        self.launch_power = 50.0  # Mid-range power
        self.prev_space_held = False

        self.packets.clear()
        self.viruses.clear()
        self.particles.clear()

        self.virus_spawn_timer = 0.0
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- 1. Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held, shift_held)

        # --- 2. Update Game Logic ---
        reward += self._update_packets()
        self._update_viruses()
        self._update_particles()
        reward += self._spawn_viruses()

        # --- 3. Calculate Rewards & Termination ---
        current_level = (self.viruses_destroyed // 100) + 1
        if current_level > self.level:
            self.level = current_level
            reward += 10.0 # Level survival bonus

        terminated = self.system_health <= 0 or self.steps >= self.MAX_STEPS
        truncated = False # This env does not truncate based on time limit in the same way as `TimeLimit` wrapper
        if self.steps >= self.MAX_STEPS:
            terminated = True # Reached max steps
            
        if self.system_health <= 0:
            terminated = True
            reward = -100.0 # Penalty for core destruction
        
        if terminated:
            self.game_over = True
            if self.system_health > 0 and self.steps >= self.MAX_STEPS: # Survived all steps
                reward = 100.0


        # --- 4. Return Gym Tuple ---
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Angle adjustment
        if movement == 1:  # Up
            self.aim_angle -= 0.05
        elif movement == 2:  # Down
            self.aim_angle += 0.05
        self.aim_angle = np.clip(self.aim_angle, -math.pi, 0)

        # Power adjustment
        if movement == 3:  # Left
            self.launch_power -= 2
        elif movement == 4:  # Right
            self.launch_power += 2
        self.launch_power = np.clip(self.launch_power, 10, 100)

        # Launch packet on space PRESS
        if space_held and not self.prev_space_held:
            self._spawn_packet()
            # sfx: player_shoot.wav
        self.prev_space_held = space_held

    def _update_packets(self):
        reward = 0
        packets_to_remove = []
        for i, packet in enumerate(self.packets):
            packet["pos"] += packet["vel"]
            packet["trail"].append(tuple(packet["pos"]))

            # Check for off-screen
            if not (0 <= packet["pos"][0] <= self.SCREEN_WIDTH and 0 <= packet["pos"][1] <= self.SCREEN_HEIGHT):
                packets_to_remove.append(i)
                reward -= 0.1 # Penalty for missed shot
                continue
            
            # Check for collision with viruses
            hit = False
            for virus in self.viruses:
                if np.linalg.norm(packet["pos"] - virus["pos"]) < virus["size"]:
                    hit = True
                    virus["health"] -= packet["damage"]
                    reward += 0.1 # Reward for hitting a virus
                    self._create_particles(packet["pos"], 10, self.VIRUS_COLORS[virus["type"]], 1.5)
                    # sfx: virus_hit.wav
                    if virus["health"] <= 0:
                        reward += 1.0 # Bonus for destroying virus
                        self.score += 10
                        self.viruses_destroyed += 1
                        self._create_particles(virus["pos"], 30, self.VIRUS_COLORS[virus["type"]], 3.0)
                        virus["is_dead"] = True
                        # sfx: virus_destroy.wav
                    break # Packet can only hit one virus
            
            if hit:
                packets_to_remove.append(i)

        # Remove packets that hit or went off-screen
        for i in sorted(packets_to_remove, reverse=True):
            del self.packets[i]
        return reward

    def _update_viruses(self):
        viruses_to_remove = []
        for i, virus in enumerate(self.viruses):
            if virus["is_dead"]:
                viruses_to_remove.append(i)
                continue
            
            # Move towards core
            direction = self.core_pos - virus["pos"]
            dist = np.linalg.norm(direction)
            if dist > 1:
                virus["pos"] += (direction / dist) * virus["speed"]
            
            # Check for collision with core
            if dist < 20: # Core radius
                self.system_health -= virus["damage"]
                self.system_health = max(0, self.system_health)
                self._create_particles(self.core_pos, 40, self.COLOR_CORE_DANGER, 4.0)
                viruses_to_remove.append(i)
                # sfx: core_damage.wav
        
        # Remove dead or collided viruses
        for i in sorted(viruses_to_remove, reverse=True):
            del self.viruses[i]

    def _update_particles(self):
        particles_to_remove = []
        for i, p in enumerate(self.particles):
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                particles_to_remove.append(i)
        
        for i in sorted(particles_to_remove, reverse=True):
            del self.particles[i]

    def _spawn_viruses(self):
        spawn_interval = max(0.2, 1.0 - (self.steps * 0.0002))
        self.virus_spawn_timer += 1 / self.TARGET_FPS

        if self.virus_spawn_timer > spawn_interval:
            self.virus_spawn_timer = 0
            
            # Determine spawn position
            edge = self.np_random.integers(4)
            if edge == 0: # Top
                pos = np.array([self.np_random.uniform(0, self.SCREEN_WIDTH), -20.0])
            elif edge == 1: # Bottom
                pos = np.array([self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 20.0])
            elif edge == 2: # Left
                pos = np.array([-20.0, self.np_random.uniform(0, self.SCREEN_HEIGHT)])
            else: # Right
                pos = np.array([self.SCREEN_WIDTH + 20.0, self.np_random.uniform(0, self.SCREEN_HEIGHT)])

            # Determine virus type based on level
            virus_type = "RED"
            rand_val = self.np_random.random()
            if self.level >= 15 and rand_val < 0.15:
                virus_type = "YELLOW"
            elif self.level >= 10 and rand_val < 0.3:
                virus_type = "PURPLE"
            elif self.level >= 5 and rand_val < 0.4:
                virus_type = "BLUE"

            base_speed = min(5.0, 1.0 + (self.steps * 0.001))
            
            virus_stats = {
                "RED":    {"health": 10, "speed_mult": 1.0, "damage": 10, "size": 8},
                "BLUE":   {"health": 10, "speed_mult": 1.5, "damage": 10, "size": 7},
                "PURPLE": {"health": 30, "speed_mult": 0.8, "damage": 15, "size": 10},
                "YELLOW": {"health": 5,  "speed_mult": 1.2, "damage": 20, "size": 9}, # Not implemented: explosive
            }
            stats = virus_stats[virus_type]

            new_virus = {
                "pos": pos,
                "type": virus_type,
                "health": stats["health"],
                "max_health": stats["health"],
                "speed": base_speed * stats["speed_mult"],
                "damage": stats["damage"],
                "size": stats["size"],
                "color": self.VIRUS_COLORS[virus_type],
                "is_dead": False,
            }
            self.viruses.append(new_virus)
        return 0 # Spawning itself gives no reward

    def _spawn_packet(self):
        power_normalized = self.launch_power / 100.0
        speed = 4.0 + power_normalized * 8.0
        damage = 10
        
        vel = np.array([math.cos(self.aim_angle), math.sin(self.aim_angle)], dtype=float) * speed
        
        new_packet = {
            "pos": self.core_pos.copy(),
            "vel": vel,
            "damage": damage,
            "trail": deque(maxlen=15)
        }
        self.packets.append(new_packet)

    def _create_particles(self, pos, count, color, speed_mult):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "lifespan": self.np_random.integers(15, 30),
                "color": color,
                "size": self.np_random.uniform(1, 3)
            })

    def _get_observation(self):
        # --- Render all game elements to the screen surface ---
        self._render_background()
        self._render_core()
        self._render_packets()
        self._render_viruses()
        self._render_particles()
        self._render_reticle()
        self._render_ui()

        # --- Convert to numpy array (EXACT format required) ---
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "system_health": self.system_health,
            "level": self.level,
            "viruses_on_screen": len(self.viruses),
        }
    
    # --- Rendering Methods ---

    def _draw_glow_circle(self, surface, pos, radius, color, glow_color, glow_layers=5):
        x, y = int(pos[0]), int(pos[1])
        for i in range(glow_layers, 0, -1):
            alpha = 150 * (1 - (i / glow_layers))
            glow_radius = int(radius + i * 2)
            if glow_radius > 0:
                pygame.gfxdraw.filled_circle(surface, x, y, glow_radius, (*glow_color, int(alpha)))
                pygame.gfxdraw.aacircle(surface, x, y, glow_radius, (*glow_color, int(alpha)))
        
        pygame.gfxdraw.filled_circle(surface, x, y, int(radius), color)
        pygame.gfxdraw.aacircle(surface, x, y, int(radius), color)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        for i in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT), 1)
        for i in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i), 1)

    def _render_core(self):
        health_ratio = self.system_health / 100.0
        core_color = [
            int(self.COLOR_CORE_DANGER[i] + (self.COLOR_CORE[i] - self.COLOR_CORE_DANGER[i]) * health_ratio)
            for i in range(3)
        ]
        self._draw_glow_circle(self.screen, self.core_pos, 20, tuple(core_color), self.COLOR_CORE)

    def _render_packets(self):
        for packet in self.packets:
            # Trail
            for i, p_trail in enumerate(packet["trail"]):
                alpha = int(255 * (i / len(packet["trail"])))
                color = (*self.COLOR_PACKET, alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(p_trail[0]), int(p_trail[1]), 2, color)
            # Head
            self._draw_glow_circle(self.screen, packet["pos"], 4, self.COLOR_PACKET, (200, 255, 200))
    
    def _render_viruses(self):
        for virus in self.viruses:
            pos_int = (int(virus["pos"][0]), int(virus["pos"][1]))
            # Body
            self._draw_glow_circle(self.screen, virus["pos"], virus["size"], virus["color"], virus["color"])
            # Health bar
            if virus["health"] < virus["max_health"]:
                health_ratio = virus["health"] / virus["max_health"]
                bar_width = 20
                bar_height = 4
                bar_x = pos_int[0] - bar_width // 2
                bar_y = pos_int[1] - virus["size"] - 8
                pygame.draw.rect(self.screen, (100, 0, 0), (bar_x, bar_y, bar_width, bar_height))
                pygame.draw.rect(self.screen, (0, 200, 0), (bar_x, bar_y, int(bar_width * health_ratio), bar_height))

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 30.0))
            alpha = max(0, min(255, alpha))
            color = (*p["color"], alpha)
            size = int(p["size"] * (p["lifespan"] / 30.0))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), size, color)

    def _render_reticle(self):
        # Aiming line
        length = 40 + self.launch_power * 0.4
        end_pos = self.core_pos + np.array([math.cos(self.aim_angle), math.sin(self.aim_angle)]) * length
        pygame.draw.aaline(self.screen, self.COLOR_PLAYER_RETICLE, self.core_pos, end_pos, 2)
        # Power arc
        power_angle = self.launch_power / 100.0 * (math.pi / 2)
        arc_rect = pygame.Rect(self.core_pos[0] - 30, self.core_pos[1] - 30, 60, 60)
        pygame.draw.arc(self.screen, self.COLOR_PLAYER_RETICLE, arc_rect, -self.aim_angle - power_angle / 2, -self.aim_angle + power_angle / 2, 2)

    def _render_ui(self):
        # Top-left info
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        level_text = self.font_small.render(f"LEVEL: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (10, 30))
        
        # Top-right info
        angle_deg = abs(math.degrees(self.aim_angle))
        angle_text = self.font_small.render(f"ANGLE: {angle_deg:.0f}", True, self.COLOR_TEXT)
        self.screen.blit(angle_text, (self.SCREEN_WIDTH - 120, 10))
        power_text = self.font_small.render(f"POWER: {self.launch_power:.0f}", True, self.COLOR_TEXT)
        self.screen.blit(power_text, (self.SCREEN_WIDTH - 120, 30))

        # Bottom health bar
        health_bar_width = self.SCREEN_WIDTH * 0.6
        health_bar_height = 20
        health_bar_x = (self.SCREEN_WIDTH - health_bar_width) / 2
        health_bar_y = self.SCREEN_HEIGHT - 35
        
        health_ratio = self.system_health / 100.0
        current_health_width = health_bar_width * health_ratio
        
        health_color = [
            int(self.COLOR_CORE_DANGER[i] + (self.COLOR_CORE[i] - self.COLOR_CORE_DANGER[i]) * health_ratio)
            for i in range(3)
        ]

        pygame.draw.rect(self.screen, (50, 50, 50), (health_bar_x, health_bar_y, health_bar_width, health_bar_height))
        pygame.draw.rect(self.screen, tuple(health_color), (health_bar_x, health_bar_y, current_health_width, health_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (health_bar_x, health_bar_y, health_bar_width, health_bar_height), 2)
        
        health_label = self.font_large.render("SYSTEM INTEGRITY", True, self.COLOR_TEXT_HEADLINE)
        self.screen.blit(health_label, (health_bar_x + health_bar_width / 2 - health_label.get_width() / 2, health_bar_y - 2))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


# --- Example Usage ---
if __name__ == '__main__':
    # Set a non-dummy driver for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # Controls: Arrow keys for angle/power, Space to shoot
    
    obs, info = env.reset()
    done = False
    
    # Use a separate screen for human rendering
    pygame.display.set_caption("Cyber Defense")
    human_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    action = [0, 0, 0] # [movement, space, shift]
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        
        # Movement
        action[0] = 0 # No-op
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # Buttons
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the human-visible screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(GameEnv.TARGET_FPS)
        
    print(f"Game Over! Final Info: {info}")
    env.close()