import gymnasium as gym
import os
import pygame
import numpy as np
import math
import random
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
from collections import deque
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

class GameEnv(gym.Env):
    """
    Pilot a salvage ship through asteroid fields, dismantling derelict spacecraft 
    for valuable resources while fending off pirates.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    game_description = "Pilot a salvage ship through asteroid fields, dismantling derelict spacecraft for valuable resources while fending off pirates."
    user_guide = "Controls: Use arrow keys (↑↓←→) to move. Press space to fire your selected tool. Press shift to cycle between your laser and salvage beam."
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Critical Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 5000
        self.PLAYER_SIZE = 12
        self.PLAYER_ACCEL = 0.6
        self.PLAYER_DRAG = 0.96
        self.PLAYER_MAX_SPEED = 6
        self.PLAYER_HULL_MAX = 100
        self.PIRATE_SPAWN_INTERVAL_INITIAL = 200
        self.PIRATE_DIFFICULTY_INTERVAL = 500

        self.RESOURCE_TYPES = {
            "common": {"color": (100, 255, 100), "value": 1, "hull": 10},
            "uncommon": {"color": (100, 150, 255), "value": 5, "hull": 20},
            "rare": {"color": (200, 100, 255), "value": 20, "hull": 40},
        }
        self.WIN_CONDITION = {"rare": 3, "uncommon": 2}
        
        # --- Colors & Fonts ---
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_GLOW = (50, 255, 150, 50)
        self.COLOR_PIRATE = (255, 80, 80)
        self.COLOR_DERELICT = (120, 130, 140)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_BG = (30, 40, 60, 180)
        self.COLOR_HULL_BAR = (100, 220, 100)
        self.COLOR_HULL_BAR_BG = (220, 100, 100)

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 48)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.player_pos = None
        self.player_vel = None
        self.player_hull = 0
        self.player_facing = None
        self.inventory = {}
        self.tools = ["laser", "salvage_beam"]
        self.selected_tool_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.laser_cooldown = 0
        self.pirate_spawn_timer = 0
        self.pirate_spawn_interval = self.PIRATE_SPAWN_INTERVAL_INITIAL
        self.pirate_projectile_speed = 2.5

        self.stars = []
        self.derelicts = []
        self.pirates = []
        self.projectiles = []
        self.resources = []
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_hull = self.PLAYER_HULL_MAX
        self.player_facing = pygame.Vector2(0, -1) # Start facing up
        
        self.inventory = {res_type: 0 for res_type in self.RESOURCE_TYPES}
        self.selected_tool_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.laser_cooldown = 0

        self.pirate_spawn_timer = self.PIRATE_SPAWN_INTERVAL_INITIAL
        self.pirate_spawn_interval = self.PIRATE_SPAWN_INTERVAL_INITIAL
        self.pirate_projectile_speed = 2.5
        
        self.stars = [
            (random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT), random.uniform(0.5, 1.5))
            for _ in range(150)
        ]
        self.derelicts = [self._create_derelict() for _ in range(3)]
        self.pirates = []
        self.projectiles = []
        self.resources = []
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Input & Player Actions ---
        self._handle_player_input(movement, space_held, shift_held)

        # --- Update Game Logic ---
        self._update_player()
        reward += self._update_pirates()
        self._update_projectiles()
        reward += self._update_resources()
        self._update_particles()
        
        # --- Collision Detection ---
        reward += self._handle_collisions()

        # --- Spawning & Difficulty Scaling ---
        self._update_spawners()
        self._update_difficulty()

        # --- Check Termination Conditions ---
        terminated = False
        if self.player_hull <= 0:
            terminated = True
            self.game_over = True
            reward = -100.0
            self.win_message = "SHIP DESTROYED"
            self._create_explosion(self.player_pos, self.COLOR_PLAYER, 100)
        
        if self._check_win_condition():
            terminated = True
            self.game_over = True
            reward = 100.0
            self.win_message = "OBJECTIVE COMPLETE"

        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
            terminated = True # Per Gymnasium API, terminated should be True if truncated
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # region Update Logic
    def _handle_player_input(self, movement, space_held, shift_held):
        # Movement
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1 # Up
        elif movement == 2: move_vec.y = 1 # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1 # Right
        
        if move_vec.length() > 0:
            self.player_vel += move_vec.normalize() * self.PLAYER_ACCEL
            self.player_facing = move_vec.normalize()
            # Thruster particles
            if self.steps % 3 == 0:
                p_pos = self.player_pos - self.player_facing * self.PLAYER_SIZE
                p_vel = -self.player_facing * random.uniform(1, 2) + pygame.Vector2(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
                self.particles.append(Particle(p_pos, p_vel, 15, (200, 200, 100)))

        # Cycle Tool (on press)
        if shift_held and not self.last_shift_held:
            self.selected_tool_idx = (self.selected_tool_idx + 1) % len(self.tools)
        
        # Fire Tool (on press)
        if space_held and not self.last_space_held:
            tool = self.tools[self.selected_tool_idx]
            if tool == "laser" and self.laser_cooldown <= 0:
                self._fire_laser()
            elif tool == "salvage_beam":
                self._fire_salvage_beam()

    def _update_player(self):
        if self.player_vel.length() > self.PLAYER_MAX_SPEED:
            self.player_vel.scale_to_length(self.PLAYER_MAX_SPEED)
        self.player_pos += self.player_vel
        self.player_vel *= self.PLAYER_DRAG

        # World Wrap
        self.player_pos.x %= self.WIDTH
        self.player_pos.y %= self.HEIGHT
        
        if self.laser_cooldown > 0:
            self.laser_cooldown -= 1

    def _update_pirates(self):
        reward = 0
        for pirate in self.pirates:
            dist_to_player = self.player_pos.distance_to(pirate.pos)
            if dist_to_player < pirate.aggro_radius:
                direction = (self.player_pos - pirate.pos).normalize()
                pirate.vel += direction * pirate.accel
                if pirate.fire_cooldown <= 0:
                    self._fire_pirate_laser(pirate)
            else: # Patrol
                if pirate.pos.distance_to(pirate.patrol_target) < 50:
                    pirate.patrol_target = pygame.Vector2(random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT))
                direction = (pirate.patrol_target - pirate.pos).normalize()
                pirate.vel += direction * pirate.accel * 0.5
            
            pirate.update()
        return reward

    def _update_projectiles(self):
        for p in self.projectiles:
            p.update()
        self.projectiles = [p for p in self.projectiles if 0 < p.pos.x < self.WIDTH and 0 < p.pos.y < self.HEIGHT and p.lifespan > 0]

    def _update_resources(self):
        reward = 0
        for r in self.resources:
            r.update(self.player_pos)
            if self.player_pos.distance_to(r.pos) < self.PLAYER_SIZE + r.size:
                r.collected = True
                self.inventory[r.res_type] += 1
                reward += 0.1
                if r.res_type in ["rare", "uncommon"]:
                    reward += 10 if r.res_type == "rare" else 2
                self.score += self.RESOURCE_TYPES[r.res_type]["value"]
        self.resources = [r for r in self.resources if not r.collected]
        return reward

    def _update_particles(self):
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.lifespan > 0]

    def _update_spawners(self):
        # Pirate Spawner
        self.pirate_spawn_timer -= 1
        if self.pirate_spawn_timer <= 0:
            if len(self.pirates) < 5: # Max pirates
                self.pirates.append(self._create_pirate())
            self.pirate_spawn_timer = self.pirate_spawn_interval

        # Derelict Spawner
        if len(self.derelicts) < 3:
            self.derelicts.append(self._create_derelict())

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.PIRATE_DIFFICULTY_INTERVAL == 0:
            self.pirate_spawn_interval = max(50, self.pirate_spawn_interval * 0.9)
            self.pirate_projectile_speed = min(5, self.pirate_projectile_speed + 0.05)

    def _handle_collisions(self):
        reward = 0
        projectiles_to_remove = set()
        
        for i, proj in enumerate(self.projectiles):
            if i in projectiles_to_remove: continue

            # Player projectiles
            if proj.owner == 'player':
                # vs Pirates
                for pirate in self.pirates:
                    if proj.pos.distance_to(pirate.pos) < pirate.size:
                        pirate.hull -= proj.damage
                        projectiles_to_remove.add(i)
                        self._create_explosion(proj.pos, self.COLOR_PIRATE, 5)
                        if pirate.hull <= 0:
                            self._create_explosion(pirate.pos, self.COLOR_PIRATE, 40)
                            reward += 5
                            self.score += 50
                        break
                
                # vs Derelicts (salvage beam)
                if proj.tool_type == 'salvage_beam':
                    for derelict in self.derelicts:
                        for component in derelict.components:
                            comp_pos = derelict.pos + component.offset
                            if proj.pos.distance_to(comp_pos) < component.size:
                                component.hull -= proj.damage
                                projectiles_to_remove.add(i)
                                self._create_explosion(proj.pos, self.RESOURCE_TYPES[component.res_type]['color'], 3, 0.5)
                                if component.hull <= 0:
                                    self._create_explosion(comp_pos, self.RESOURCE_TYPES[component.res_type]['color'], 15)
                                    self.resources.append(ResourceDrop(pygame.Vector2(comp_pos), component.res_type))
                                break
                        if i in projectiles_to_remove: break
            
            # Pirate projectiles vs Player
            elif proj.owner == 'pirate':
                if proj.pos.distance_to(self.player_pos) < self.PLAYER_SIZE:
                    self.player_hull -= proj.damage
                    projectiles_to_remove.add(i)
                    self._create_explosion(proj.pos, self.COLOR_PLAYER, 10)
                    reward -= 0.5
                    break
        
        self.projectiles = [p for i, p in enumerate(self.projectiles) if i not in projectiles_to_remove]
        self.pirates = [p for p in self.pirates if p.hull > 0]
        
        for derelict in self.derelicts:
            derelict.components = [c for c in derelict.components if c.hull > 0]
        self.derelicts = [d for d in self.derelicts if d.components]

        return reward
    # endregion

    # region Firing & Creation
    def _fire_laser(self):
        proj_pos = self.player_pos + self.player_facing * (self.PLAYER_SIZE + 5)
        proj_vel = self.player_facing * 8 + self.player_vel * 0.5
        self.projectiles.append(Projectile(proj_pos, proj_vel, 'player', damage=10, color=self.COLOR_PLAYER, tool_type='laser'))
        self.laser_cooldown = 10
        self._create_explosion(proj_pos, (255,255,255), 5, 1, 3)

    def _fire_salvage_beam(self):
        proj_pos = self.player_pos + self.player_facing * (self.PLAYER_SIZE + 5)
        proj_vel = self.player_facing * 6
        self.projectiles.append(Projectile(proj_pos, proj_vel, 'player', damage=5, color=(255, 255, 100), lifespan=20, tool_type='salvage_beam'))
        self._create_explosion(proj_pos, (255,255,100), 5, 1, 3)

    def _fire_pirate_laser(self, pirate):
        direction = (self.player_pos - pirate.pos).normalize()
        proj_pos = pirate.pos + direction * (pirate.size + 5)
        proj_vel = direction * self.pirate_projectile_speed
        self.projectiles.append(Projectile(proj_pos, proj_vel, 'pirate', damage=10, color=self.COLOR_PIRATE))
        pirate.fire_cooldown = 120 # 4 seconds at 30fps

    def _create_derelict(self):
        pos = pygame.Vector2(random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT))
        while self.player_pos is not None and pos.distance_to(self.player_pos) < 200:
            pos = pygame.Vector2(random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT))
        return Derelict(pos, self.RESOURCE_TYPES)

    def _create_pirate(self):
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top': pos = pygame.Vector2(random.randint(0, self.WIDTH), -20)
        elif edge == 'bottom': pos = pygame.Vector2(random.randint(0, self.WIDTH), self.HEIGHT + 20)
        elif edge == 'left': pos = pygame.Vector2(-20, random.randint(0, self.HEIGHT))
        else: pos = pygame.Vector2(self.WIDTH + 20, random.randint(0, self.HEIGHT))
        return Pirate(pos)

    def _create_explosion(self, pos, color, num_particles, speed_mult=1, lifespan_mult=1):
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) * speed_mult
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = random.randint(15, 30) * lifespan_mult
            self.particles.append(Particle(pygame.Vector2(pos), vel, lifespan, color))
    # endregion

    # region State & Rendering
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
            "hull": self.player_hull,
            "inventory": self.inventory,
            "win_condition_met": self._check_win_condition(),
        }

    def _check_win_condition(self):
        return all(self.inventory.get(res, 0) >= self.WIN_CONDITION[res] for res in self.WIN_CONDITION)

    def _render_game(self):
        # Stars
        for x, y, b in self.stars:
            pygame.draw.circle(self.screen, (b * 100, b * 100, b * 120), (x, y), b)
        
        # Entities (render order matters)
        for derelict in self.derelicts: derelict.draw(self.screen, self.COLOR_DERELICT)
        for resource in self.resources: resource.draw(self.screen)
        for particle in self.particles: particle.draw(self.screen)
        for pirate in self.pirates: pirate.draw(self.screen, self.COLOR_PIRATE)
        for proj in self.projectiles: proj.draw(self.screen)

        # Player
        if self.player_hull > 0:
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), self.PLAYER_SIZE + 5, self.COLOR_PLAYER_GLOW)
            # Ship body
            p1 = self.player_pos + self.player_facing * self.PLAYER_SIZE
            p2 = self.player_pos + self.player_facing.rotate(-140) * self.PLAYER_SIZE
            p3 = self.player_pos + self.player_facing.rotate(140) * self.PLAYER_SIZE
            pygame.gfxdraw.aapolygon(self.screen, [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)], self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)], self.COLOR_PLAYER)

    def _render_ui(self):
        # UI Background Panel
        panel_rect = pygame.Rect(5, 5, self.WIDTH - 10, 65)
        pygame.gfxdraw.box(self.screen, panel_rect, self.COLOR_UI_BG)

        # Hull Bar
        hull_ratio = max(0, self.player_hull / self.PLAYER_HULL_MAX)
        hull_bar_width = 200
        hull_bar_rect = pygame.Rect(15, 15, hull_bar_width * hull_ratio, 15)
        hull_bar_bg_rect = pygame.Rect(15, 15, hull_bar_width, 15)
        pygame.draw.rect(self.screen, self.COLOR_HULL_BAR_BG, hull_bar_bg_rect)
        pygame.draw.rect(self.screen, self.COLOR_HULL_BAR, hull_bar_rect)
        hull_text = self.font_medium.render(f"HULL: {int(self.player_hull)}%", True, self.COLOR_UI_TEXT)
        self.screen.blit(hull_text, (20, 35))

        # Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - 15 - score_text.get_width(), 15))

        # Tool Selection
        tool_name = self.tools[self.selected_tool_idx].replace("_", " ").upper()
        tool_text = self.font_medium.render(f"TOOL: {tool_name}", True, self.COLOR_UI_TEXT)
        self.screen.blit(tool_text, (self.WIDTH - 15 - tool_text.get_width(), 40))

        # Inventory / Win Condition
        inv_x = 240
        for i, (res_type, needed) in enumerate(self.WIN_CONDITION.items()):
            color = self.RESOURCE_TYPES[res_type]["color"]
            owned = self.inventory.get(res_type, 0)
            text = f"{res_type.upper()}: {owned}/{needed}"
            res_text = self.font_medium.render(text, True, color)
            self.screen.blit(res_text, (inv_x, 15 + i * 25))
        
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            msg_surf = self.font_large.render(self.win_message, True, self.COLOR_UI_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(msg_surf, msg_rect)

    def close(self):
        pygame.quit()
    # endregion

# region Helper Classes
class Particle:
    def __init__(self, pos, vel, lifespan, color):
        self.pos = pos
        self.vel = vel
        self.lifespan = lifespan
        self.max_lifespan = lifespan
        self.color = color

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1

    def draw(self, screen):
        alpha = int(255 * (self.lifespan / self.max_lifespan))
        temp_color = (*self.color, alpha)
        size = max(1, int(2 * (self.lifespan / self.max_lifespan)))
        # Note: pygame.draw.circle does not support alpha on non-SRCALPHA surfaces.
        # This will draw a solid color, but won't crash.
        pygame.draw.circle(screen, self.color, (int(self.pos.x), int(self.pos.y)), size)

class Projectile:
    def __init__(self, pos, vel, owner, damage, color, lifespan=60, tool_type=None):
        self.pos = pos
        self.vel = vel
        self.owner = owner
        self.damage = damage
        self.color = color
        self.lifespan = lifespan
        self.tool_type = tool_type

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1

    def draw(self, screen):
        end_pos = self.pos - self.vel.normalize() * 5
        pygame.draw.line(screen, self.color, (int(self.pos.x), int(self.pos.y)), (int(end_pos.x), int(end_pos.y)), 2)

class ResourceDrop:
    def __init__(self, pos, res_type):
        self.pos = pos
        self.res_type = res_type
        self.color = GameEnv.RESOURCE_TYPES[res_type]["color"]
        self.vel = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * 0.5
        self.size = 5
        self.collected = False
        self.attract_radius = 80

    def update(self, player_pos):
        if player_pos.distance_to(self.pos) < self.attract_radius:
            direction = (player_pos - self.pos).normalize()
            self.vel = self.vel * 0.9 + direction * 0.5
        else:
            self.vel *= 0.98
        self.pos += self.vel

    def draw(self, screen):
        pygame.gfxdraw.filled_circle(screen, int(self.pos.x), int(self.pos.y), self.size, (*self.color, 150))
        pygame.gfxdraw.aacircle(screen, int(self.pos.x), int(self.pos.y), self.size, self.color)

class Pirate:
    def __init__(self, pos):
        self.pos = pos
        self.vel = pygame.Vector2(0, 0)
        self.size = 10
        self.hull = 30
        self.accel = 0.1
        self.drag = 0.98
        self.max_speed = 2
        self.fire_cooldown = random.randint(60, 180)
        self.aggro_radius = 250
        self.patrol_target = pygame.Vector2(random.randint(0, 640), random.randint(0, 400))

    def update(self):
        if self.vel.length() > self.max_speed:
            self.vel.scale_to_length(self.max_speed)
        self.pos += self.vel
        self.vel *= self.drag
        self.fire_cooldown -= 1
        self.pos.x %= 640
        self.pos.y %= 400

    def draw(self, screen, color):
        p1 = self.pos + self.vel.normalize() * self.size if self.vel.length() > 0 else self.pos + pygame.Vector2(0, -1) * self.size
        p2 = self.pos + self.vel.normalize().rotate(-120) * self.size if self.vel.length() > 0 else self.pos + pygame.Vector2(0, -1).rotate(-120) * self.size
        p3 = self.pos + self.vel.normalize().rotate(120) * self.size if self.vel.length() > 0 else self.pos + pygame.Vector2(0, -1).rotate(120) * self.size
        pygame.gfxdraw.aapolygon(screen, [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)], color)
        pygame.gfxdraw.filled_polygon(screen, [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)], color)

class Derelict:
    def __init__(self, pos, resource_types):
        self.pos = pos
        self.components = []
        num_components = random.randint(2, 4)
        for _ in range(num_components):
            offset = pygame.Vector2(random.uniform(-30, 30), random.uniform(-30, 30))
            size = random.randint(8, 15)
            res_type = random.choices(list(resource_types.keys()), weights=[0.6, 0.3, 0.1], k=1)[0]
            hull = resource_types[res_type]["hull"] * random.uniform(0.8, 1.2)
            self.components.append(DerelictComponent(offset, size, res_type, hull))

    def draw(self, screen, color):
        for comp in self.components:
            pos = self.pos + comp.offset
            pygame.draw.rect(screen, color, (pos.x - comp.size, pos.y - comp.size, comp.size*2, comp.size*2), 2)

class DerelictComponent:
    def __init__(self, offset, size, res_type, hull):
        self.offset = offset
        self.size = size
        self.res_type = res_type
        self.hull = hull
# endregion

if __name__ == '__main__':
    # --- Example Usage & Manual Play ---
    # This block is for manual testing and visualization.
    # It will not be executed by the test harness.
    # To run, you might need to unset the SDL_VIDEODRIVER dummy variable.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Salvage Run")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Transpose for Pygame display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            print(f"Info: {info}")
            total_reward = 0
            obs, info = env.reset()
            pygame.time.wait(2000) # Pause before reset

        clock.tick(env.metadata["render_fps"])
        
    env.close()