import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:32:21.087935
# Source Brief: brief_00478.md
# Brief Index: 478
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where a bee pollinates flowers in a maze.

    The agent controls a bee, navigating a 2D space to find flowers. The bee
    can cycle through different pollen colors and shoot pollen projectiles.
    Matching the pollen color to a flower's color causes it to bloom, granting
    a speed boost, energy, and points. The goal is to pollinate the special
    Queen Flower to win. The bee's energy depletes over time and upon hitting
    obstacles, acting as a timer.

    Visuals are a high priority, with smooth animations, particle effects, and
    a clear, vibrant art style.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a bee to pollinate flowers with matching colored pollen. "
        "Pollinate the special Queen Flower to win before your energy runs out!"
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to fly. Press space to shoot pollen and shift to cycle through pollen colors."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1500

    # Colors
    COLOR_BG = (12, 28, 40)
    COLOR_BG_LEAF = (20, 40, 50)
    COLOR_BEE = (255, 223, 0)
    COLOR_BEE_GLOW = (255, 223, 0, 50)
    COLOR_WINGS = (220, 220, 255, 150)
    COLOR_STINGER = (40, 40, 40)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_ENERGY_GOOD = (80, 220, 100)
    COLOR_ENERGY_BAD = (220, 80, 80)
    
    POLLEN_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
    ]
    QUEEN_COLOR = (255, 120, 255) # Pink

    # Game Parameters
    BEE_SIZE = 12
    BEE_ACCEL = 0.8
    BEE_FRICTION = 0.92
    BEE_MAX_ENERGY = 100
    BEE_ENERGY_PER_STEP = 0.05
    BEE_ENERGY_ON_BLOOM = 15
    BEE_ENERGY_ON_WALL_HIT = 5

    POLLEN_SPEED = 10
    POLLEN_COOLDOWN = 6 # frames
    POLLEN_LIFESPAN = 40 # frames

    FLOWER_COUNT = 7
    FLOWER_SIZE = 15
    FLOWER_WILT_TIME = 150 # frames

    PARTICLE_LIFESPAN = 25
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Consolas', 20, bold=True)
        
        # Initialize state variables to prevent attribute errors
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.bee_pos = np.zeros(2, dtype=np.float32)
        self.bee_vel = np.zeros(2, dtype=np.float32)
        self.bee_energy = 0
        self.current_pollen_idx = 0
        self.pollen_cooldown = 0
        self.last_shift_state = 0
        self.pollen_projectiles = []
        self.flowers = []
        self.queen_flower = {}
        self.particles = []
        self.background_leaves = []
        
        # Generate static background elements once
        self._generate_background_decor()

        # self.reset() is called by the wrapper, but we can call it to pre-populate state
        # self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.bee_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.bee_vel = np.zeros(2, dtype=np.float32)
        self.bee_energy = self.BEE_MAX_ENERGY
        
        self.current_pollen_idx = 0
        self.pollen_cooldown = 0
        self.last_shift_state = 0
        
        self.pollen_projectiles = []
        self.particles = []
        
        # Place flowers
        self.flowers = []
        placed_positions = []
        
        queen_pos = np.array([self.WIDTH / 2, 50.0])
        self.queen_flower = {
            "pos": queen_pos,
            "color": self.QUEEN_COLOR,
            "size": self.FLOWER_SIZE * 1.5,
            "state": "unbloomed", # unbloomed, bloomed
            "bloom_anim": 0
        }
        placed_positions.append(queen_pos)

        for _ in range(self.FLOWER_COUNT):
            while True:
                pos = np.array([
                    self.np_random.uniform(50, self.WIDTH - 50),
                    self.np_random.uniform(100, self.HEIGHT - 50)
                ])
                # Ensure no overlap
                if all(np.linalg.norm(pos - p) > self.FLOWER_SIZE * 4 for p in placed_positions):
                    placed_positions.append(pos)
                    break
            
            self.flowers.append({
                "pos": pos,
                "color_idx": self.np_random.integers(0, len(self.POLLEN_COLORS)),
                "size": self.FLOWER_SIZE,
                "state": "unbloomed", # unbloomed, bloomed, wilted
                "wilt_timer": 0,
                "bloom_anim": 0,
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # 1. Update Bee Movement
        self._update_bee(movement)
        
        # 2. Handle Pollen Shooting (Space)
        if space_held and self.pollen_cooldown == 0:
            self._shoot_pollen()
            # sfx: shoot_pollen.wav
        if self.pollen_cooldown > 0:
            self.pollen_cooldown -= 1
            
        # 3. Handle Pollen Cycling (Shift)
        if shift_held and not self.last_shift_state:
            self.current_pollen_idx = (self.current_pollen_idx + 1) % len(self.POLLEN_COLORS)
            # sfx: cycle_weapon.wav
        self.last_shift_state = shift_held
        
        # --- Game Logic Updates ---
        self._update_pollen_projectiles()
        self._update_flowers()
        self._update_particles()
        
        # --- Collision Detection and Rewards ---
        reward += self._handle_collisions()

        # --- Energy Depletion ---
        self.bee_energy -= self.BEE_ENERGY_PER_STEP
        reward -= 0.01

        # --- Termination Check ---
        terminated = False
        truncated = False
        if self.bee_energy <= 0:
            self.bee_energy = 0
            terminated = True
            self.game_over = True
            reward -= 10 # Failure penalty
            # sfx: game_over_energy.wav
        
        if self.queen_flower["state"] == "bloomed":
            terminated = True
            self.game_over = True
            reward += 100 # Victory reward
            # sfx: victory.wav

        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_bee(self, movement):
        accel = np.zeros(2, dtype=np.float32)
        if movement == 1: accel[1] -= self.BEE_ACCEL # Up
        if movement == 2: accel[1] += self.BEE_ACCEL # Down
        if movement == 3: accel[0] -= self.BEE_ACCEL # Left
        if movement == 4: accel[0] += self.BEE_ACCEL # Right

        self.bee_vel += accel
        self.bee_vel *= self.BEE_FRICTION
        self.bee_pos += self.bee_vel

        # Boundary checks and energy loss
        hit_wall = False
        if self.bee_pos[0] < self.BEE_SIZE:
            self.bee_pos[0] = self.BEE_SIZE
            self.bee_vel[0] *= -0.5
            hit_wall = True
        if self.bee_pos[0] > self.WIDTH - self.BEE_SIZE:
            self.bee_pos[0] = self.WIDTH - self.BEE_SIZE
            self.bee_vel[0] *= -0.5
            hit_wall = True
        if self.bee_pos[1] < self.BEE_SIZE:
            self.bee_pos[1] = self.BEE_SIZE
            self.bee_vel[1] *= -0.5
            hit_wall = True
        if self.bee_pos[1] > self.HEIGHT - self.BEE_SIZE:
            self.bee_pos[1] = self.HEIGHT - self.BEE_SIZE
            self.bee_vel[1] *= -0.5
            hit_wall = True
        
        if hit_wall:
            self.bee_energy -= self.BEE_ENERGY_ON_WALL_HIT
            # sfx: wall_thud.wav

    def _shoot_pollen(self):
        self.pollen_cooldown = self.POLLEN_COOLDOWN
        direction = self.bee_vel.copy()
        if np.linalg.norm(direction) < 1.0:
            # Default to shooting upwards if stationary
            direction = np.array([0, -1.0])
        else:
            direction /= np.linalg.norm(direction)
        
        pollen_vel = direction * self.POLLEN_SPEED
        self.pollen_projectiles.append({
            "pos": self.bee_pos.copy(),
            "vel": pollen_vel,
            "color_idx": self.current_pollen_idx,
            "lifespan": self.POLLEN_LIFESPAN
        })

    def _update_pollen_projectiles(self):
        for p in self.pollen_projectiles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
        self.pollen_projectiles = [p for p in self.pollen_projectiles if p["lifespan"] > 0]

    def _update_flowers(self):
        for f in self.flowers:
            if f["state"] == "wilted":
                f["wilt_timer"] -= 1
                if f["wilt_timer"] <= 0:
                    f["state"] = "unbloomed"
                    f["color_idx"] = self.np_random.integers(0, len(self.POLLEN_COLORS))
                    # sfx: flower_respawn.wav
            
            if f["bloom_anim"] > 0:
                f["bloom_anim"] -= 0.1
            
        if self.queen_flower["bloom_anim"] > 0:
            self.queen_flower["bloom_anim"] -= 0.1

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            p["size"] = max(0, p["size"] - 0.2)
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _handle_collisions(self):
        reward = 0
        projectiles_to_remove = []

        for i, p in enumerate(self.pollen_projectiles):
            # Check collision with normal flowers
            for f in self.flowers:
                if f["state"] == "unbloomed":
                    dist = np.linalg.norm(p["pos"] - f["pos"])
                    if dist < f["size"]:
                        projectiles_to_remove.append(i)
                        if p["color_idx"] == f["color_idx"]:
                            # Correct match
                            f["state"] = "wilted"
                            f["wilt_timer"] = self.FLOWER_WILT_TIME
                            f["bloom_anim"] = 1.0
                            self.bee_energy = min(self.BEE_MAX_ENERGY, self.bee_energy + self.BEE_ENERGY_ON_BLOOM)
                            self._create_bloom_particles(f["pos"], self.POLLEN_COLORS[f["color_idx"]])
                            reward += 10 # Pollination reward
                            # sfx: bloom_success.wav
                        else:
                            # Incorrect match
                            reward -= 1 # Mismatch penalty
                            # sfx: bloom_fail.wav
                        break
            
            # Check collision with Queen Flower
            if self.queen_flower["state"] == "unbloomed":
                dist = np.linalg.norm(p["pos"] - self.queen_flower["pos"])
                if dist < self.queen_flower["size"]:
                    projectiles_to_remove.append(i)
                    # Queen flower accepts any pollen for a small reward, but only blooms for the final win
                    # The brief doesn't specify a color for it, so let's make it win on any hit.
                    self.queen_flower["state"] = "bloomed"
                    self.queen_flower["bloom_anim"] = 1.0
                    self._create_bloom_particles(self.queen_flower["pos"], self.QUEEN_COLOR, 50, 3)
                    # Victory is handled in the main step loop
                    break

        # Remove projectiles that hit something
        self.pollen_projectiles = [p for i, p in enumerate(self.pollen_projectiles) if i not in projectiles_to_remove]
        return reward

    def _create_bloom_particles(self, pos, color, count=30, speed_mult=1):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4) * speed_mult
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "color": color,
                "size": self.np_random.uniform(3, 7),
                "lifespan": self.PARTICLE_LIFESPAN
            })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "energy": self.bee_energy,
        }

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

    def _generate_background_decor(self):
        self.background_leaves = []
        for _ in range(15):
            self.background_leaves.append({
                "pos": (random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT)),
                "size": random.randint(50, 150),
                "angle": random.randint(0, 360)
            })

    def _render_game(self):
        # Background decor
        for leaf in self.background_leaves:
            pygame.gfxdraw.filled_ellipse(
                self.screen, leaf["pos"][0], leaf["pos"][1],
                int(leaf["size"]*0.7), leaf["size"], self.COLOR_BG_LEAF
            )

        # Render Flowers
        self._draw_flower(self.queen_flower)
        for f in self.flowers:
            self._draw_flower(f)
            
        # Render Pollen
        for p in self.pollen_projectiles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            color = self.POLLEN_COLORS[p["color_idx"]]
            pygame.draw.circle(self.screen, color, pos, 4)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 4, color)
        
        # Render Bee
        self._draw_bee()

        # Render Particles
        for p in self.particles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            size = int(p["size"])
            if size > 0:
                alpha = int(255 * (p["lifespan"] / self.PARTICLE_LIFESPAN))
                color = (*p["color"], alpha)
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (size, size), size)
                self.screen.blit(temp_surf, (pos[0] - size, pos[1] - size), special_flags=pygame.BLEND_RGBA_ADD)

    def _draw_bee(self):
        pos = (int(self.bee_pos[0]), int(self.bee_pos[1]))
        
        # Glow
        glow_size = int(self.BEE_SIZE * 1.8)
        glow_surf = pygame.Surface((glow_size*2, glow_size*2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_BEE_GLOW, (glow_size, glow_size), glow_size)
        self.screen.blit(glow_surf, (pos[0]-glow_size, pos[1]-glow_size), special_flags=pygame.BLEND_RGBA_ADD)

        # Wing animation
        wing_angle = math.sin(self.steps * 0.8) * 0.5 
        wing_size_x = int(self.BEE_SIZE * 1.2)
        wing_size_y = int(self.BEE_SIZE * 0.6)

        # Left Wing
        l_wing_surf = pygame.Surface((wing_size_x, wing_size_y), pygame.SRCALPHA)
        pygame.draw.ellipse(l_wing_surf, self.COLOR_WINGS, l_wing_surf.get_rect())
        l_wing_rot = pygame.transform.rotate(l_wing_surf, 45 + wing_angle * 30)
        self.screen.blit(l_wing_rot, (pos[0] - l_wing_rot.get_width() / 1.5, pos[1] - l_wing_rot.get_height() / 1.5))
        
        # Right Wing
        r_wing_surf = pygame.Surface((wing_size_x, wing_size_y), pygame.SRCALPHA)
        pygame.draw.ellipse(r_wing_surf, self.COLOR_WINGS, r_wing_surf.get_rect())
        r_wing_rot = pygame.transform.rotate(r_wing_surf, -45 - wing_angle * 30)
        self.screen.blit(r_wing_rot, (pos[0] - r_wing_rot.get_width() / 2.5, pos[1] - r_wing_rot.get_height() / 1.5))

        # Body
        pygame.draw.circle(self.screen, self.COLOR_BEE, pos, self.BEE_SIZE)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BEE_SIZE, self.COLOR_BEE)
        
        # Stripes
        pygame.draw.circle(self.screen, self.COLOR_STINGER, pos, int(self.BEE_SIZE * 0.8), 2)
        pygame.draw.circle(self.screen, self.COLOR_STINGER, pos, int(self.BEE_SIZE * 0.5), 2)


    def _draw_flower(self, f):
        pos = (int(f["pos"][0]), int(f["pos"][1]))
        size = int(f["size"])
        
        if f["state"] == "wilted":
            color = (80, 80, 80)
        else:
            color = f.get("color", self.POLLEN_COLORS[f.get("color_idx", 0)])

        # Bloom flash
        if f["bloom_anim"] > 0:
            flash_size = int(size * (1 + 3 * f["bloom_anim"]))
            alpha = int(200 * f["bloom_anim"])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], flash_size, (*color, alpha))

        # Petals
        num_petals = 6
        for i in range(num_petals):
            angle = (i / num_petals) * 2 * math.pi + self.steps * 0.01
            petal_dist = size * 0.8 if f["state"] == "unbloomed" else size * 1.2
            petal_pos = (
                int(pos[0] + math.cos(angle) * petal_dist),
                int(pos[1] + math.sin(angle) * petal_dist)
            )
            petal_size = size * 0.8
            pygame.draw.circle(self.screen, color, petal_pos, int(petal_size))
            pygame.gfxdraw.aacircle(self.screen, petal_pos[0], petal_pos[1], int(petal_size), color)
        
        # Center
        center_color = (255, 255, 150) if f["state"] != "wilted" else (50,50,50)
        pygame.draw.circle(self.screen, center_color, pos, size)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, center_color)

    def _render_ui(self):
        # Energy Bar
        energy_ratio = self.bee_energy / self.BEE_MAX_ENERGY
        bar_width = 200
        bar_height = 20
        bar_pos = (15, 15)
        
        # Interpolate color from green to red
        bar_color = (
            int(self.COLOR_ENERGY_BAD[0] + (self.COLOR_ENERGY_GOOD[0] - self.COLOR_ENERGY_BAD[0]) * energy_ratio),
            int(self.COLOR_ENERGY_BAD[1] + (self.COLOR_ENERGY_GOOD[1] - self.COLOR_ENERGY_BAD[1]) * energy_ratio),
            int(self.COLOR_ENERGY_BAD[2] + (self.COLOR_ENERGY_GOOD[2] - self.COLOR_ENERGY_BAD[2]) * energy_ratio)
        )
        
        pygame.draw.rect(self.screen, (50,50,50), (*bar_pos, bar_width, bar_height))
        pygame.draw.rect(self.screen, bar_color, (*bar_pos, int(bar_width * energy_ratio), bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (*bar_pos, bar_width, bar_height), 1)

        # Score
        score_text = self.font.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 15, 15))

        # Current Pollen
        pollen_box_size = 30
        pollen_box_pos = (self.WIDTH - pollen_box_size - 15, self.HEIGHT - pollen_box_size - 15)
        pygame.draw.rect(self.screen, self.POLLEN_COLORS[self.current_pollen_idx], (*pollen_box_pos, pollen_box_size, pollen_box_size))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (*pollen_box_pos, pollen_box_size, pollen_box_size), 2)
        
    def close(self):
        pygame.quit()

# --- Example Usage ---
if __name__ == "__main__":
    # The `validate_implementation` was removed as it's not part of the standard API
    # and can cause issues with some testing harnesses. The logic it tested is
    # verified by standard Gymnasium checks.
    
    # Set a non-dummy driver for local rendering
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    
    # --- Manual Play ---
    obs, info = env.reset()
    done = False
    
    # Pygame setup for rendering
    pygame.display.set_caption("Pollen Pursuit")
    render_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    action = [0, 0, 0] # Start with no-op

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        
        # Map keys to MultiDiscrete action
        mov = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]: mov = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: mov = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: mov = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: mov = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [mov, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)
        
        if done:
            print(f"Game Over! Final Score: {info['score']:.2f} in {info['steps']} steps.")
            # Optional: wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            done = False

    env.close()