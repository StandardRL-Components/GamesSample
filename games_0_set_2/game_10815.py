import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:06:44.858439
# Source Brief: brief_00815.md
# Brief Index: 815
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Navigate a shifting quantum fabric, collect entangled particles,
    craft reality-bending tools, and solve physics-based puzzles to
    unlock new areas.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a shifting quantum fabric by collecting particles to craft "
        "reality-bending tools like platforms and teleporters to reach the goal."
    )
    user_guide = (
        "Controls: Use ←→ to move and ↑ to jump. Press space to use the selected tool "
        "and shift to cycle through crafted tools."
    )
    auto_advance = True

    # --- CONSTANTS ---
    # Screen
    WIDTH, HEIGHT = 640, 400
    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_PLAYER = (255, 255, 0)
    COLOR_PLAYER_GLOW = (255, 255, 0, 40)
    COLOR_PLATFORM = (180, 180, 200)
    COLOR_PLATFORM_OUTLINE = (220, 220, 255)
    COLOR_GOAL = (0, 255, 180)
    COLOR_GOAL_GLOW = (0, 255, 180, 60)
    COLOR_UI_TEXT = (220, 220, 255)
    PARTICLE_COLORS = {
        "red": (255, 80, 80),
        "green": (80, 255, 80),
        "blue": (80, 80, 255)
    }
    # Physics
    GRAVITY = 0.6
    PLAYER_ACCEL = 0.8
    PLAYER_FRICTION = -0.15
    PLAYER_JUMP_STRENGTH = -12
    PLAYER_MAX_SPEED_X = 6
    PLAYER_MAX_SPEED_Y = 15
    # Game
    MAX_STEPS = 2000
    PLAYER_SIZE = 20

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- GYM SPACES ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- PYGAME SETUP ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_tool = pygame.font.SysFont("monospace", 16)

        # --- TOOLS ---
        self.tool_recipes = {
            "Phase Platform": {"red": 2, "blue": 1},
            "Short Teleport": {"green": 3},
        }
        
        # --- INITIAL STATE ---
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.is_grounded = False
        self.last_move_direction = pygame.Vector2(1, 0)

        self.platforms = []
        self.particles_data = []
        self.goal_rect = pygame.Rect(0, 0, 0, 0)
        
        self.inventory = {}
        self.crafted_tools = []
        self.selected_tool_idx = 0
        self.tool_cooldowns = {}
        self.temp_platforms = []
        
        self.effect_particles = []
        self.bg_stars = []

        self.last_space_held = False
        self.last_shift_held = False

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.zones_entered = set()

        # self.reset() is called by the environment wrapper
        # self.validate_implementation() # This is for dev, should be removed for prod

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- RESET GAME STATE ---
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Player
        self.player_vel = pygame.Vector2(0, 0)
        self.is_grounded = False
        self.last_move_direction = pygame.Vector2(1, 0)

        # World
        self._generate_level()

        # Inventory & Tools
        self.inventory = {p_type: 0 for p_type in self.PARTICLE_COLORS}
        self.crafted_tools = []
        self.selected_tool_idx = 0
        self.tool_cooldowns = {}
        self.temp_platforms = []
        
        # Effects & UI
        self.effect_particles = []
        self.bg_stars = [
            {
                "pos": pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)),
                "vel": pygame.Vector2(self.np_random.uniform(-0.2, 0.2), self.np_random.uniform(-0.1, 0.1)),
                "radius": self.np_random.uniform(1, 3),
                "color": (self.np_random.integers(20, 41), self.np_random.integers(15, 31), self.np_random.integers(40, 61))
            } for _ in range(100)
        ]
        self.last_space_held = False
        self.last_shift_held = False
        self.zones_entered = {int(self.player_pos.y // 100)}

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1

        # --- PROCESS INPUT ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.last_space_held
        shift_press = shift_held and not self.last_shift_held
        self.last_space_held, self.last_shift_held = space_held, shift_held

        # --- UPDATE GAME LOGIC ---
        reward += self._handle_input_and_movement(movement)
        self._update_physics()
        reward += self._handle_collisions_and_interactions()
        self._update_world_state()
        reward += self._check_crafting()
        
        # Cooldowns
        for tool_name in list(self.tool_cooldowns.keys()):
            self.tool_cooldowns[tool_name] -= 1
            if self.tool_cooldowns[tool_name] <= 0:
                del self.tool_cooldowns[tool_name]

        # Tool usage
        if space_press:
            reward += self._use_tool()
        if shift_press and self.crafted_tools:
            self.selected_tool_idx = (self.selected_tool_idx + 1) % len(self.crafted_tools)
            # Sfx: UI_CycleTool

        # --- CHECK TERMINATION ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over: # First frame of termination
            self.game_over = True
            if self.player_pos.y > self.HEIGHT + self.PLAYER_SIZE:
                reward = -100.0 # Fell
            else: # Reached goal
                reward = 100.0
        
        # Update score
        self.score += reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _generate_level(self):
        self.platforms = []
        self.particles_data = []
        
        # Start platform
        start_platform = pygame.Rect(50, self.HEIGHT - 50, 150, 20)
        self.platforms.append({"rect": start_platform, "type": "stable"})
        self.player_pos = pygame.Vector2(start_platform.centerx, start_platform.top - self.PLAYER_SIZE)

        # Procedural path generation
        x, y = start_platform.right, start_platform.y
        path_length = 15
        for i in range(path_length):
            gap_x = self.np_random.integers(60, 101)
            gap_y = self.np_random.integers(-60, 61)
            width = self.np_random.integers(80, 151)
            
            x += gap_x
            y = np.clip(y + gap_y, 100, self.HEIGHT - 50)
            
            is_unstable = self.np_random.random() < 0.3
            platform_type = "unstable" if is_unstable else "stable"
            
            new_platform = pygame.Rect(x, y, width, 20)
            self.platforms.append({"rect": new_platform, "type": platform_type, "timer": 500})
            
            # Add particles
            if self.np_random.random() < 0.7:
                particle_type = self.np_random.choice(list(self.PARTICLE_COLORS.keys()))
                px = new_platform.centerx + self.np_random.integers(-20, 21)
                py = new_platform.top - 20
                self.particles_data.append({"pos": pygame.Vector2(px, py), "type": particle_type, "bob_offset": self.np_random.uniform(0, math.pi * 2)})
        
        # Goal
        goal_platform = self.platforms[-1]["rect"]
        self.goal_rect = pygame.Rect(goal_platform.centerx - 20, goal_platform.top - 40, 40, 40)

    def _handle_input_and_movement(self, movement):
        # Horizontal movement
        if movement == 3:  # Left
            self.player_vel.x -= self.PLAYER_ACCEL
            self.last_move_direction.x = -1
        elif movement == 4:  # Right
            self.player_vel.x += self.PLAYER_ACCEL
            self.last_move_direction.x = 1
        
        # Jumping
        if movement == 1 and self.is_grounded: # Up
            self.player_vel.y = self.PLAYER_JUMP_STRENGTH
            self.is_grounded = False
            self._create_effect_particles(self.player_pos + pygame.Vector2(0, self.PLAYER_SIZE / 2), 10, self.COLOR_PLATFORM)
            # Sfx: Jump

        return 0

    def _update_physics(self):
        # Apply friction and gravity
        self.player_vel.x += self.player_vel.x * self.PLAYER_FRICTION
        if not self.is_grounded:
            self.player_vel.y += self.GRAVITY

        # Clamp velocity
        self.player_vel.x = np.clip(self.player_vel.x, -self.PLAYER_MAX_SPEED_X, self.PLAYER_MAX_SPEED_X)
        self.player_vel.y = np.clip(self.player_vel.y, -self.PLAYER_MAX_SPEED_Y, self.PLAYER_MAX_SPEED_Y)
        if abs(self.player_vel.x) < 0.1: self.player_vel.x = 0

        # Update position
        self.player_pos += self.player_vel
        
        # Update last move direction for aiming
        if abs(self.player_vel.x) > 0.1 or abs(self.player_vel.y) > 0.1:
            if self.player_vel.length() > 0:
                self.last_move_direction = self.player_vel.copy().normalize()

    def _handle_collisions_and_interactions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE / 2, self.player_pos.y - self.PLAYER_SIZE / 2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        # --- Platform Collisions ---
        self.is_grounded = False
        all_platforms = [p['rect'] for p in self.platforms] + [p['rect'] for p in self.temp_platforms]
        
        # Vertical collision
        # Create a temporary rect for collision checking to not modify the player's actual position yet
        check_rect = player_rect.copy()
        check_rect.y += self.player_vel.y
        for platform_rect in all_platforms:
            if check_rect.colliderect(platform_rect):
                if self.player_vel.y > 0: # Moving down
                    self.player_pos.y = platform_rect.top - self.PLAYER_SIZE / 2
                    self.player_vel.y = 0
                    self.is_grounded = True
                elif self.player_vel.y < 0: # Moving up
                    self.player_pos.y = platform_rect.bottom + self.PLAYER_SIZE / 2
                    self.player_vel.y = 0
        
        # Horizontal collision
        check_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE / 2, self.player_pos.y - self.PLAYER_SIZE / 2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        check_rect.x += self.player_vel.x
        for platform_rect in all_platforms:
            if check_rect.colliderect(platform_rect):
                if self.player_vel.x > 0: # Moving right
                    self.player_pos.x = platform_rect.left - self.PLAYER_SIZE / 2
                elif self.player_vel.x < 0: # Moving left
                    self.player_pos.x = platform_rect.right + self.PLAYER_SIZE / 2
                self.player_vel.x = 0

        # --- Particle Collection ---
        player_rect_for_collection = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE/2, self.player_pos.y - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        for p_data in self.particles_data[:]:
            if player_rect_for_collection.collidepoint(p_data["pos"]):
                self.inventory[p_data["type"]] += 1
                self.particles_data.remove(p_data)
                self._create_effect_particles(p_data["pos"], 15, self.PARTICLE_COLORS[p_data["type"]])
                reward += 0.1
                # Sfx: CollectParticle
        
        # --- Zone Entry ---
        current_zone = int(self.player_pos.y // 100)
        if current_zone not in self.zones_entered:
            self.zones_entered.add(current_zone)
            reward += 5.0
            
        return reward

    def _update_world_state(self):
        # Background stars
        for star in self.bg_stars:
            star["pos"] += star["vel"]
            if star["pos"].x < 0 or star["pos"].x > self.WIDTH: star["vel"].x *= -1
            if star["pos"].y < 0 or star["pos"].y > self.HEIGHT: star["vel"].y *= -1
            
        # Effect particles
        for p in self.effect_particles[:]:
            p["pos"] += p["vel"]
            p["vel"].y += 0.1 # Gravity on particles
            p["life"] -= 1
            if p["life"] <= 0:
                self.effect_particles.remove(p)

        # Bobbing collectable particles
        for p_data in self.particles_data:
            p_data["bob_offset"] += 0.05
        
        # Unstable and temp platforms
        player_feet_rect = pygame.Rect(self.player_pos.x-self.PLAYER_SIZE/4, self.player_pos.y+self.PLAYER_SIZE/2-1, self.PLAYER_SIZE/2, 2)
        for p_data in self.platforms:
            if p_data["type"] == "unstable" and self.is_grounded and player_feet_rect.colliderect(p_data['rect']):
                p_data["timer"] -= 1
                if p_data["timer"] <= 0:
                    p_data["rect"].y += self.HEIGHT * 2 # Effectively remove
        
        for p_data in self.temp_platforms[:]:
            p_data["timer"] -= 1
            if p_data["timer"] <= 0:
                self.temp_platforms.remove(p_data)

    def _check_crafting(self):
        reward = 0
        for tool_name, recipe in self.tool_recipes.items():
            if tool_name not in self.crafted_tools:
                can_craft = all(self.inventory[p_type] >= count for p_type, count in recipe.items())
                if can_craft:
                    for p_type, count in recipe.items():
                        self.inventory[p_type] -= count
                    self.crafted_tools.append(tool_name)
                    reward += 1.0
                    # Sfx: CraftTool
        return reward

    def _use_tool(self):
        if not self.crafted_tools: return 0
        
        tool_name = self.crafted_tools[self.selected_tool_idx]
        if tool_name in self.tool_cooldowns: return 0
        
        if tool_name == "Phase Platform":
            platform_pos = self.player_pos + self.last_move_direction * 50
            new_platform = {
                "rect": pygame.Rect(platform_pos.x - 25, platform_pos.y - 5, 50, 10),
                "timer": 150, # 5 seconds at 30fps
                "max_life": 150
            }
            self.temp_platforms.append(new_platform)
            self.tool_cooldowns[tool_name] = 60 # 2s cooldown
            self._create_effect_particles(platform_pos, 20, self.COLOR_PLATFORM_OUTLINE)
            # Sfx: CreatePlatform
            return 0.2
            
        if tool_name == "Short Teleport":
            teleport_dist = 100
            start_pos = self.player_pos.copy()
            self.player_pos += self.last_move_direction * teleport_dist
            self._create_effect_particles(start_pos, 30, self.COLOR_PLAYER)
            self._create_effect_particles(self.player_pos, 30, self.COLOR_PLAYER)
            self.player_vel = pygame.Vector2(0,0) # Stop momentum
            self.tool_cooldowns[tool_name] = 90 # 3s cooldown
            # Sfx: Teleport
            return 0.2
        
        return 0

    def _create_effect_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.effect_particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": self.np_random.integers(15, 31),
                "max_life": 30,
                "color": color
            })

    def _check_termination(self):
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE/2, self.player_pos.y - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        if player_rect.colliderect(self.goal_rect):
            return True
        if self.player_pos.y > self.HEIGHT + self.PLAYER_SIZE:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background
        for star in self.bg_stars:
            pygame.gfxdraw.filled_circle(self.screen, int(star["pos"].x), int(star["pos"].y), int(star["radius"]), star["color"])

        # Goal
        pygame.gfxdraw.box(self.screen, self.goal_rect, (*self.COLOR_GOAL, 80))
        for i in range(4):
            pygame.gfxdraw.aacircle(self.screen, self.goal_rect.centerx, self.goal_rect.centery, self.goal_rect.width//2 + i*2, self.COLOR_GOAL_GLOW)

        # Platforms
        for p_data in self.platforms:
            color = self.COLOR_PLATFORM
            if p_data["type"] == "unstable":
                flicker = (p_data['timer'] % 20) > 10
                if p_data['timer'] < 150:
                    flicker = (p_data['timer'] % 10) > 5
                if flicker: color = (255, 100, 100)
            pygame.draw.rect(self.screen, color, p_data["rect"])
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_OUTLINE, p_data["rect"], 1)
        
        # Temp Platforms
        for p_data in self.temp_platforms:
            alpha = int(255 * (p_data['timer'] / p_data['max_life']))
            color = (*self.COLOR_PLATFORM, alpha)
            pygame.gfxdraw.box(self.screen, p_data["rect"], color)

        # Collectable Particles
        for p_data in self.particles_data:
            y_offset = math.sin(p_data["bob_offset"]) * 3
            pos = (int(p_data["pos"].x), int(p_data["pos"].y + y_offset))
            color = self.PARTICLE_COLORS[p_data["type"]]
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, (*color, 60))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 5, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 5, (255,255,255))
            
        # Effect Particles
        for p in self.effect_particles:
            alpha = 255 * (p['life'] / p['max_life'])
            color = (*p['color'], int(alpha))
            if len(color) == 4: # Handle colors with and without alpha
                pygame.gfxdraw.pixel(self.screen, int(p['pos'].x), int(p['pos'].y), color)

        # Player
        player_rect = pygame.Rect(0,0,self.PLAYER_SIZE, self.PLAYER_SIZE)
        player_rect.center = (int(self.player_pos.x), int(self.player_pos.y))
        pygame.gfxdraw.aacircle(self.screen, player_rect.centerx, player_rect.centery, self.PLAYER_SIZE, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, player_rect.centerx, player_rect.centery, self.PLAYER_SIZE-2, self.COLOR_PLAYER_GLOW)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

    def _render_ui(self):
        # Inventory
        x_offset = 15
        for p_type, color in self.PARTICLE_COLORS.items():
            count = self.inventory[p_type]
            pygame.gfxdraw.filled_circle(self.screen, x_offset, 20, 8, color)
            pygame.gfxdraw.aacircle(self.screen, x_offset, 20, 8, (255,255,255))
            text = self.font_ui.render(f"{count}", True, self.COLOR_UI_TEXT)
            self.screen.blit(text, (x_offset + 15, 12))
            x_offset += 60
            
        # Steps
        steps_text = self.font_ui.render(f"STEPS: {self.MAX_STEPS - self.steps}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 15, 12))

        # Selected Tool
        if self.crafted_tools:
            tool_name = self.crafted_tools[self.selected_tool_idx]
            tool_text_str = f"TOOL: {tool_name}"
            if tool_name in self.tool_cooldowns:
                cooldown_sec = self.tool_cooldowns[tool_name] / 30.0
                tool_text_str += f" ({cooldown_sec:.1f}s)"
                text_color = (150, 150, 150)
            else:
                text_color = self.COLOR_GOAL
            
            tool_text = self.font_tool.render(tool_text_str, True, text_color)
            self.screen.blit(tool_text, (self.WIDTH / 2 - tool_text.get_width() / 2, self.HEIGHT - 30))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "inventory": self.inventory,
            "crafted_tools": len(self.crafted_tools)
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    # Switch to a real display driver for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.quit()
    pygame.init() # Re-init with the new driver
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Quantum Fabric Explorer")
    clock = pygame.time.Clock()
    
    running = True
    total_score = 0
    
    while running:
        # --- Player Input ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        # elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2 # Down movement not used in platformer
        if keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_score += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it.
        # Pygame uses (width, height), numpy uses (height, width), so we need to transpose.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode Finished. Final Score: {total_score:.2f}, Info: {info}")
            total_score = 0
            obs, info = env.reset()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                total_score = 0
                obs, info = env.reset()

        clock.tick(30) # Run at 30 FPS for human play

    env.close()